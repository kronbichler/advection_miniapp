// This file is part of the advection_miniapp repository and subject to the
// LGPL license. See the LICENSE file in the top level directory of this
// project for details.

// Program for time integration of the advection problem, realizing an
// implicit backward Euler integration with local solvers
// Author: Martin Kronbichler, Technical University of Munich, 2014-2022
//
// This program shares many similarities with the step-67 tutorial program of
// deal.II, see https://dealii.org/developer/doxygen/deal.II/step_67.html ,
// but it implements a simpler equation and is therefore ideal for learning
// about matrix-free evaluators.

// Compared to the main program advection_solver.cc, this program implements
// a variant with variable transport speed derived from an analytical
// expression. This file intentionally duplicates most of the other
// program to keep all implementation in a single file; for sustainable
// software design it would be advisable to share common code between the two
// cases.

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_idr.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/point_value_history.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iomanip>
#include <iostream>



namespace DGAdvection
{
  using namespace dealii;

  // The dimension can be selected to be 1, 2, 3 (it is a C++ template
  // argument, so different code gets compiled in 1D/2D/3D); for the advection
  // speed of a rotating vortex that is set here, only dimension 2 is allowed
  const unsigned int dimension = 2;

  // The polynomial degree can be selected between 0 and any reasonable number
  // (around 30), depending on the dimension and the mesh size
  const unsigned int fe_degree = 5;

  // The time step size is controlled via this parameter as
  // dt = courant_number * min_h / transport_norm
  const double courant_number = 4;

  // Diffusion coefficient
  const double diffusion = 1e-4;

  // 0: central flux, 1: classical upwind flux (= Lax-Friedrichs)
  const double flux_alpha = 1.0;

  // The final simulation time
  const double FINAL_TIME = 8.0;

  // Frequency of output
  const double output_tick = 0.1;

  // Whether to mesh the domain with Cartesian mesh elements or with curved
  // elements (more memory transfer -> slower)
  enum class MeshType
  {
    cartesian,
    deformed_cartesian
  };
  constexpr MeshType mesh_type = MeshType::cartesian;

  // Whether to set periodic boundary conditions on the domain (needs periodic
  // solution as well)
  const bool periodic = true;

  // Switch to change between a conservative formulation of the advection term
  // (factor 0) or a skew-symmetric one (factor 0.5)
  const double factor_skew = 0.5;

  // Switch to enable Gauss-Lobatto quadrature (true) or Gauss quadrature
  // (false)
  const bool use_gl_quad = false;

  // Switch to enable Gauss--Lobatto quadrature for the inverse mass
  // matrix. If false, use Gauss quadrature
  const bool use_gl_quad_mass = false;

  // Enable or disable writing of result files for visualization with ParaView
  // or VisIt
  const bool print_vtu = true;



  // Analytical solution of the problem
  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double time = 0.)
      : Function<dim>(1, time)
    {}

    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return value<double>(p);
    }

    template <typename Number>
    Number
    value(const Point<dim, Number> &p) const
    {
      return std::exp(
        -400. * ((p[0] - 0.5) * (p[0] - 0.5) + (p[1] - 0.75) * (p[1] - 0.75)));
    }
  };



  template <int dim>
  class TransportSpeed
  {
  public:
    TransportSpeed(const double time = 0.0)
      : time(time)
    {}

    template <typename Number>
    Tensor<1, dim, Number>
    value(const Point<dim, Number> &) const
    {
      Tensor<1, dim, Number> transport;
      transport[0] = 1.9;
      if (dim > 1)
        transport[1] = -0.7;
      if (dim > 2)
        transport[2] = 1.1;
      return transport;
    }

  private:
    const double time;
  };



  // Description of curved mesh
  template <int dim>
  class DeformedCubeManifold : public ChartManifold<dim, dim, dim>
  {
  public:
    DeformedCubeManifold(const double       left,
                         const double       right,
                         const double       deformation,
                         const unsigned int frequency = 1)
      : left(left)
      , right(right)
      , deformation(deformation)
      , frequency(frequency)
    {}

    Point<dim>
    push_forward(const Point<dim> &chart_point) const override
    {
      double sinval = deformation;
      for (unsigned int d = 0; d < dim; ++d)
        sinval *= std::sin(frequency * numbers::PI * (chart_point(d) - left) /
                           (right - left));
      Point<dim> space_point;
      for (unsigned int d = 0; d < dim; ++d)
        space_point(d) = chart_point(d) + sinval;
      return space_point;
    }

    Point<dim>
    pull_back(const Point<dim> &space_point) const override
    {
      Point<dim> x = space_point;
      Point<dim> one;
      for (unsigned int d = 0; d < dim; ++d)
        one(d) = 1.;

      // Newton iteration to solve the nonlinear equation given by the point
      Tensor<1, dim> sinvals;
      for (unsigned int d = 0; d < dim; ++d)
        sinvals[d] =
          std::sin(frequency * numbers::PI * (x(d) - left) / (right - left));

      double sinval = deformation;
      for (unsigned int d = 0; d < dim; ++d)
        sinval *= sinvals[d];
      Tensor<1, dim> residual = space_point - x - sinval * one;
      unsigned int   its      = 0;
      while (residual.norm() > 1e-12 && its < 100)
        {
          Tensor<2, dim> jacobian;
          for (unsigned int d = 0; d < dim; ++d)
            jacobian[d][d] = 1.;
          for (unsigned int d = 0; d < dim; ++d)
            {
              double sinval_der = deformation * frequency / (right - left) *
                                  numbers::PI *
                                  std::cos(frequency * numbers::PI *
                                           (x(d) - left) / (right - left));
              for (unsigned int e = 0; e < dim; ++e)
                if (e != d)
                  sinval_der *= sinvals[e];
              for (unsigned int e = 0; e < dim; ++e)
                jacobian[e][d] += sinval_der;
            }

          x += invert(jacobian) * residual;

          for (unsigned int d = 0; d < dim; ++d)
            sinvals[d] = std::sin(frequency * numbers::PI * (x(d) - left) /
                                  (right - left));

          sinval = deformation;
          for (unsigned int d = 0; d < dim; ++d)
            sinval *= sinvals[d];
          residual = space_point - x - sinval * one;
          ++its;
        }
      AssertThrow(residual.norm() < 1e-12,
                  ExcMessage("Newton for point did not converge."));
      return x;
    }

    std::unique_ptr<Manifold<dim>>
    clone() const override
    {
      return std::make_unique<DeformedCubeManifold<dim>>(left,
                                                         right,
                                                         deformation,
                                                         frequency);
    }

  private:
    const double       left;
    const double       right;
    const double       deformation;
    const unsigned int frequency;
  };



  template <int dim, int fe_degree, typename Number = double>
  class CellwisePreconditionerFDM
  {
  public:
    static constexpr unsigned int n = fe_degree + 1;
    using vcomplex                  = std::complex<VectorizedArray<Number>>;

    CellwisePreconditionerFDM() = default;

    void
    vmult(VectorizedArray<Number>                        *in_out_array,
          const std::array<AlignedVector<vcomplex>, dim> &eigenvectors,
          const std::array<AlignedVector<vcomplex>, dim> &inverse_eigenvectors,
          const std::array<AlignedVector<vcomplex>, dim> &eigenvalues,
          const Number                                    inv_dt) const
    {
      // copy from real to complex vector
      for (unsigned int i = 0; i < data_array.size(); ++i)
        {
          data_array[i].real(in_out_array[i]);
          data_array[i].imag(VectorizedArray<Number>());
        }

      using Eval = internal::
        EvaluatorTensorProduct<internal::evaluate_general, dim, n, n, vcomplex>;
      // apply V M^{-1}
      Eval::template apply<0, false, false>(inverse_eigenvectors[0].data(),
                                            data_array.data(),
                                            data_array.data());
      if (dim > 1)
        Eval::template apply<1, false, false>(inverse_eigenvectors[1].data(),
                                              data_array.data(),
                                              data_array.data());
      if (dim > 2)
        Eval::template apply<2, false, false>(inverse_eigenvectors[2].data(),
                                              data_array.data(),
                                              data_array.data());

      for (unsigned int i2 = 0, c = 0; i2 < (dim > 2 ? n : 1); ++i2)
        for (unsigned int i1 = 0; i1 < (dim > 1 ? n : 1); ++i1)
          for (unsigned int i0 = 0; i0 < n; ++i0, ++c)
            {
              std::array<unsigned int, 3> indices{{i0, i1, i2}};
              vcomplex diagonal_element = make_vectorized_array<Number>(inv_dt);
              for (unsigned int d = 0; d < dim; ++d)
                diagonal_element += eigenvalues[d][indices[d]];
              data_array[c] /= diagonal_element;
            }

      // apply V^{-1}
      Eval::template apply<0, false, false>(eigenvectors[0].data(),
                                            data_array.data(),
                                            data_array.data());
      if (dim > 1)
        Eval::template apply<1, false, false>(eigenvectors[1].data(),
                                              data_array.data(),
                                              data_array.data());
      if (dim > 2)
        Eval::template apply<2, false, false>(eigenvectors[2].data(),
                                              data_array.data(),
                                              data_array.data());

      // copy back to real vector
      for (unsigned int i = 0; i < data_array.size(); ++i)
        in_out_array[i] = data_array[i].real();
    }

  private:
    mutable std::array<vcomplex, Utilities::pow(n, dim)> data_array;
  };



  // Implementation of the advection operation
  template <int dim, int fe_degree>
  class AdvectionOperation
  {
  public:
    using Number = double;

    AdvectionOperation()
      : time(0.)
      , time_step(0.)
      , computing_times(4)
    {}

    void
    reinit(const DoFHandler<dim> &dof_handler);

    void
    initialize_dof_vector(LinearAlgebra::distributed::Vector<Number> &vec)
    {
      data.initialize_dof_vector(vec);
    }

    const MatrixFree<dim, Number> &
    get_matrix_free() const
    {
      return data;
    }

    ~AdvectionOperation()
    {
      if (computing_times[2] > 0)
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "Advection operator evaluated "
                      << (std::size_t)computing_times[2] << " times."
                      << std::endl
                      << "Time vmult (min / avg / max): ";
          Utilities::MPI::MinMaxAvg data =
            Utilities::MPI::min_max_avg(computing_times[0], MPI_COMM_WORLD);
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << data.min << " (proc_" << data.min_index << ") / "
                      << data.avg << " / " << data.max << " (proc_"
                      << data.max_index << ")" << std::endl;
          data =
            Utilities::MPI::min_max_avg(computing_times[3], MPI_COMM_WORLD);
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "Time block-Jacobi prec (min / avg / max): "
                      << data.min << " (proc_" << data.min_index << ") / "
                      << data.avg << " / " << data.max << " (proc_"
                      << data.max_index << ")" << std::endl;
        }
    }

    void
    set_time(const double current_time, const double time_step)
    {
      this->time      = current_time;
      this->time_step = time_step;
    }

    void
    vmult(LinearAlgebra::distributed::Vector<Number>       &dst,
          const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      Timer time;
      data.loop(&AdvectionOperation<dim, fe_degree>::local_apply_domain,
                &AdvectionOperation<dim, fe_degree>::local_apply_inner_face,
                &AdvectionOperation<dim, fe_degree>::local_apply_boundary_face,
                this,
                dst,
                src,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
      computing_times[0] += time.wall_time();
      ++computing_times[2];
    }

    void
    apply_inverse_mass_matrix(
      LinearAlgebra::distributed::Vector<Number> &dst) const
    {
      FEEvaluation<dim, -1, 0, 1, Number> phi(data);
      MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 1, Number>
        mass_inv(phi);

      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values(dst);
          mass_inv.apply(phi.begin_dof_values(), phi.begin_dof_values());
          phi.set_dof_values(dst);
        }
    }

    void
    compute_matrix(SparseMatrix<double> &sparse_matrix) const
    {
      AffineConstraints<Number> constraints;
      constraints.close();
      MatrixFreeTools::compute_matrix(data,
                                      constraints,
                                      sparse_matrix,
                                      &AdvectionOperation::apply_cell_eval,
                                      &AdvectionOperation::apply_face_eval,
                                      &AdvectionOperation::apply_boundary_eval,
                                      this);
    }

    void
    precondition_block_jacobi(
      LinearAlgebra::distributed::Vector<Number>       &dst,
      const LinearAlgebra::distributed::Vector<Number> &src) const;

    void
    project_initial(LinearAlgebra::distributed::Vector<Number> &dst) const;

    Tensor<1, 3>
    compute_mass_and_energy(
      const LinearAlgebra::distributed::Vector<Number> &vec) const;

  private:
    MatrixFree<dim, Number> data;
    double                  time;
    double                  time_step;

    mutable std::vector<double> computing_times;

    std::array<AlignedVector<std::complex<VectorizedArray<Number>>>, dim>
      eigenvectors, inverse_eigenvectors;
    std::array<AlignedVector<std::complex<VectorizedArray<Number>>>, dim>
      eigenvalues;

    void
    local_apply_domain(
      const MatrixFree<dim, Number>                    &data,
      LinearAlgebra::distributed::Vector<Number>       &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int>      &cell_range) const;

    void
    local_apply_inner_face(
      const MatrixFree<dim, Number>                    &data,
      LinearAlgebra::distributed::Vector<Number>       &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int>      &cell_range) const;
    void
    local_apply_boundary_face(
      const MatrixFree<dim, Number>                    &data,
      LinearAlgebra::distributed::Vector<Number>       &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int>      &cell_range) const;

    void
    apply_cell_eval(FEEvaluation<dim, -1, 0, 1, Number> &eval) const
    {
      eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

      evaluate_on_cell(eval);

      // multiply by nabla v^h(x) and sum
      eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    }

    template <typename CellEvalType>
    void
    evaluate_on_cell(CellEvalType &eval) const
    {
      const double inv_dt = time_step == 0 ? 0. : 1. / time_step;

      const Tensor<1, dim, VectorizedArray<Number>> speed =
        TransportSpeed<dim>().value(eval.quadrature_point(0));

      // loop over quadrature points and compute the local volume flux
      for (const unsigned int q : eval.quadrature_point_indices())
        {
          const auto                              u     = eval.get_value(q);
          const auto                              gradu = eval.get_gradient(q);
          Tensor<1, dim, VectorizedArray<Number>> volume_flux =
            ((-1.0 + factor_skew) * speed) * u;
          eval.submit_gradient(
            volume_flux + make_vectorized_array<Number>(diffusion) * gradu, q);
          VectorizedArray<Number> volume_val =
            ((factor_skew * speed) * gradu) + inv_dt * u;
          eval.submit_value(volume_val, q);
        }
    }

    void
    apply_face_eval(FEFaceEvaluation<dim, -1, 0, 1, Number> &eval_minus,
                    FEFaceEvaluation<dim, -1, 0, 1, Number> &eval_plus) const
    {
      eval_minus.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
      eval_plus.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

      evaluate_on_face(eval_minus, eval_plus);

      eval_minus.integrate(EvaluationFlags::values |
                           EvaluationFlags::gradients);
      eval_plus.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    }

    template <typename FaceEvalType>
    void
    evaluate_on_face(FaceEvalType &eval_minus, FaceEvalType &eval_plus) const
    {
      const VectorizedArray<Number> normal_times_speed =
        TransportSpeed<dim>().value(eval_minus.quadrature_point(0)) *
        eval_minus.normal_vector(0);
      const VectorizedArray<Number> sigmaF =
        (std::abs((eval_minus.normal_vector(0) *
                   eval_plus.inverse_jacobian(0))[dim - 1]) +
         std::abs((eval_minus.normal_vector(0) *
                   eval_plus.inverse_jacobian(0))[dim - 1])) *
        (Number)(std::max(fe_degree, 1) * (fe_degree + 1.0));

      for (const unsigned int q : eval_minus.quadrature_point_indices())
        {
          const auto u_minus = eval_minus.get_value(q);
          const auto u_plus  = eval_plus.get_value(q);

          const auto flux_times_normal_of_u_minus =
            0.5 *
            ((u_minus + u_plus) * normal_times_speed +
             flux_alpha * std::abs(normal_times_speed) * (u_minus - u_plus));
          const VectorizedArray<Number> convective_minus =
            flux_times_normal_of_u_minus -
            factor_skew * normal_times_speed * u_minus;
          const VectorizedArray<Number> convective_plus =
            -flux_times_normal_of_u_minus +
            factor_skew * normal_times_speed * u_plus;

          const VectorizedArray<Number> jump_value =
            (u_minus - u_plus) * Number(-0.5 * diffusion);
          const VectorizedArray<Number> average_derivative =
            (eval_minus.get_normal_derivative(q) +
             eval_plus.get_normal_derivative(q)) *
            Number(-0.5 * diffusion);
          const VectorizedArray<Number> viscous_value_flux =
            jump_value * sigmaF - average_derivative;

          eval_minus.submit_normal_derivative(jump_value, q);
          eval_plus.submit_normal_derivative(jump_value, q);
          eval_minus.submit_value(convective_minus - viscous_value_flux, q);
          eval_plus.submit_value(convective_plus + viscous_value_flux, q);
        }
    }

    void
    apply_boundary_eval(FEFaceEvaluation<dim, -1, 0, 1, Number> &) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    template <typename FaceEvalType>
    void
    evaluate_on_face(FaceEvalType &) const
    {}
  };



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::reinit(const DoFHandler<dim> &dof_handler)
  {
    MappingQGeneric<dim> mapping(fe_degree);
    Quadrature<1>        quadrature = QGauss<1>(fe_degree + 1);
    if (use_gl_quad)
      quadrature = QGaussLobatto<1>(fe_degree + 1);
    Quadrature<1> quadrature_mass = QGauss<1>(fe_degree + 1);
    if (use_gl_quad_mass || use_gl_quad)
      quadrature_mass = QGaussLobatto<1>(fe_degree + 1);
    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.overlap_communication_computation = false;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points |
       update_values);
    additional_data.mapping_update_flags_inner_faces =
      (update_JxW_values | update_normal_vectors | update_quadrature_points |
       update_values | update_gradients);
    additional_data.mapping_update_flags_boundary_faces =
      (update_JxW_values | update_normal_vectors | update_quadrature_points |
       update_values | update_gradients);

    AffineConstraints<double> dummy;
    dummy.close();
    data.reinit(mapping,
                std::vector<const DoFHandler<dim> *>{&dof_handler},
                std::vector<const AffineConstraints<double> *>{&dummy},
                std::vector<Quadrature<1>>{{quadrature, quadrature_mass}},
                additional_data);

    QGauss<1>    gauss_quad(fe_degree + 1);
    FE_DGQ<1>    fe_1d(fe_degree);
    const double h = dof_handler.begin_active()->vertex(1)[0] -
                     dof_handler.begin_active()->vertex(0)[0];
    constexpr unsigned int n = fe_degree + 1;
    const Tensor<1, dim>   transport_speed =
      TransportSpeed<dim>().value(Point<dim>());
    for (unsigned int d = 0; d < dim; ++d)
      {
        LAPACKFullMatrix<double> deriv_matrix(n, n);
        LAPACKFullMatrix<double> mass_matrix(n, n);
        mass_matrix.set_property(LAPACKSupport::symmetric);
        for (unsigned int q = 0; q < n; ++q)
          {
            for (unsigned int i = 0; i < n; ++i)
              for (unsigned int j = 0; j < n; ++j)
                deriv_matrix(i, j) +=
                  (diffusion / h * fe_1d.shape_grad(i, gauss_quad.point(q)) *
                     fe_1d.shape_grad(j, gauss_quad.point(q)) -
                   transport_speed[d] *
                     fe_1d.shape_grad(i, gauss_quad.point(q))[0] *
                     fe_1d.shape_value(j, gauss_quad.point(q))) *
                  gauss_quad.weight(q);
            for (unsigned int i = 0; i < n; ++i)
              for (unsigned int j = 0; j < n; ++j)
                mass_matrix(i, j) += fe_1d.shape_value(i, gauss_quad.point(q)) *
                                     fe_1d.shape_value(j, gauss_quad.point(q)) *
                                     gauss_quad.weight(q) * h;
          }
        const double sign_advection = (transport_speed[d] > 0) ? 1.0 : -1.0;
        const double sigma          = fe_degree * (fe_degree + 1) / h;
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            deriv_matrix(i, j) +=
              (-transport_speed[d] * fe_1d.shape_value(i, Point<1>()) *
                 fe_1d.shape_value(j, Point<1>()) *
                 (0.5 - flux_alpha * 0.5 * sign_advection) +
               (0.5 * fe_1d.shape_value(i, Point<1>()) *
                  fe_1d.shape_grad(j, Point<1>())[0] / h +
                0.5 * fe_1d.shape_value(j, Point<1>()) *
                  fe_1d.shape_grad(i, Point<1>())[0] / h +
                fe_1d.shape_value(i, Point<1>()) *
                  fe_1d.shape_value(j, Point<1>()) * sigma) *
                 diffusion) +
              (transport_speed[d] * fe_1d.shape_value(i, Point<1>(1.0)) *
                 fe_1d.shape_value(j, Point<1>(1.0)) *
                 (0.5 + flux_alpha * 0.5 * sign_advection) -
               (0.5 * fe_1d.shape_value(i, Point<1>(1.0)) *
                  fe_1d.shape_grad(j, Point<1>(1.0))[0] / h +
                0.5 * fe_1d.shape_value(j, Point<1>(1.0)) *
                  fe_1d.shape_grad(i, Point<1>(1.0))[0] / h -
                fe_1d.shape_value(i, Point<1>(1.0)) *
                  fe_1d.shape_value(j, Point<1>(1.0)) * sigma) *
                 diffusion);

        if (false)
          {
            std::cout << "Derivative: " << std::endl;
            deriv_matrix.print_formatted(std::cout);
            std::cout << "Mass: " << std::endl;
            mass_matrix.print_formatted(std::cout);
          }

        mass_matrix.compute_cholesky_factorization();
        mass_matrix.solve(deriv_matrix);
        deriv_matrix.compute_eigenvalues(true, false);

        eigenvalues[d].resize(n);
        for (unsigned int i = 0; i < n; ++i)
          eigenvalues[d][i] = deriv_matrix.eigenvalue(i);

        eigenvectors[d].resize(n * n);
        auto eigvecs = deriv_matrix.get_right_eigenvectors();
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            eigenvectors[d][i * n + j] = eigvecs(i, j);
        eigvecs.gauss_jordan();
        inverse_eigenvectors[d].resize(n * n);
        mass_matrix.invert();
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            {
              std::complex<double> sum = 0;
              for (unsigned int k = 0; k < n; ++k)
                sum += eigvecs(i, k) * mass_matrix(k, j);
              inverse_eigenvectors[d][i * n + j] = sum;
            }
      }
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::local_apply_domain(
    const MatrixFree<dim, Number>                    &data,
    LinearAlgebra::distributed::Vector<Number>       &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int>      &cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);

        // compute u^h(x) from src
        eval.gather_evaluate(src,
                             EvaluationFlags::values |
                               EvaluationFlags::gradients);

        evaluate_on_cell(eval);

        // multiply by nabla v^h(x) and sum
        eval.integrate_scatter(EvaluationFlags::values |
                                 EvaluationFlags::gradients,
                               dst);
      }
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::local_apply_inner_face(
    const MatrixFree<dim, Number>                    &data,
    LinearAlgebra::distributed::Vector<Number>       &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int>      &face_range) const
  {
    // On interior faces, we have two evaluators, one for the solution
    // 'u_minus' and one for the solution 'u_plus'. Note that the decision
    // about what is minus and plus is arbitrary at this point, so we must
    // assume that this can be arbitrarily oriented and we must only operate
    // with the generic quantities such as the normal vector.
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_minus(data,
                                                                          true);
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_plus(data,
                                                                         false);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_plus.reinit(face);
        eval_minus.gather_evaluate(src,
                                   EvaluationFlags::values |
                                     EvaluationFlags::gradients);
        eval_plus.gather_evaluate(src,
                                  EvaluationFlags::values |
                                    EvaluationFlags::gradients);

        evaluate_on_face(eval_minus, eval_plus);

        eval_minus.integrate_scatter(EvaluationFlags::values |
                                       EvaluationFlags::gradients,
                                     dst);
        eval_plus.integrate_scatter(EvaluationFlags::values |
                                      EvaluationFlags::gradients,
                                    dst);
      }
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::local_apply_boundary_face(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &,
    const LinearAlgebra::distributed::Vector<Number> &,
    const std::pair<unsigned int, unsigned int> &) const
  {
    AssertThrow(false, ExcNotImplemented());
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::project_initial(
    LinearAlgebra::distributed::Vector<Number> &dst) const
  {
    ExactSolution<dim>                                     solution(0.);
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(data);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, 1, Number>
      inverse(phi);
#if DEAL_II_VERSION_GTE(9, 3, 0)
    dst.zero_out_ghost_values();
#else
    dst.zero_out_ghosts();
#endif
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_dof_value(solution.value(phi.quadrature_point(q)), q);
        inverse.transform_from_q_points_to_basis(1,
                                                 phi.begin_dof_values(),
                                                 phi.begin_dof_values());
        phi.set_dof_values(dst);
      }
  }



  template <int dim, int fe_degree>
  Tensor<1, 3>
  AdvectionOperation<dim, fe_degree>::compute_mass_and_energy(
    const LinearAlgebra::distributed::Vector<Number> &vec) const
  {
    Tensor<1, 3>                                           mass_energy = {};
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(data);
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(vec,
                            EvaluationFlags::values |
                              EvaluationFlags::gradients);
        VectorizedArray<Number> mass   = {};
        VectorizedArray<Number> energy = {};
        VectorizedArray<Number> H1semi = {};
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            mass += phi.get_value(q) * phi.JxW(q);
            energy += phi.get_value(q) * phi.get_value(q) * phi.JxW(q);
            H1semi += (phi.get_gradient(q) * phi.get_gradient(q)) * phi.JxW(q);
          }
        for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell);
             ++v)
          {
            mass_energy[0] += mass[v];
            mass_energy[1] += energy[v];
            mass_energy[2] += H1semi[v];
          }
      }
    return Utilities::MPI::sum(mass_energy, vec.get_mpi_communicator());
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::precondition_block_jacobi(
    LinearAlgebra::distributed::Vector<Number>       &dst,
    const LinearAlgebra::distributed::Vector<Number> &src) const
  {
    Timer timer;

    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval(data);
    Vector<double> local_src(eval.dofs_per_cell *
                             VectorizedArray<Number>::size());
    Vector<double> local_dst(local_src);

    CellwisePreconditionerFDM<dim, fe_degree> precondition;

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        eval.reinit(cell);
        eval.read_dof_values(src);
        precondition.vmult(eval.begin_dof_values(),
                           eigenvectors,
                           inverse_eigenvectors,
                           eigenvalues,
                           1. / time_step);
        eval.set_dof_values(dst);
      }
    computing_times[3] += timer.wall_time();
  }



  template <typename OperatorType>
  class BlockJacobi
  {
  public:
    BlockJacobi(const OperatorType &operator_exemplar)
      : operator_exemplar(operator_exemplar)
    {}

    void
    vmult(LinearAlgebra::distributed::Vector<double>       &dst,
          const LinearAlgebra::distributed::Vector<double> &src) const
    {
      operator_exemplar.precondition_block_jacobi(dst, src);
    }

  private:
    const OperatorType &operator_exemplar;
  };



  template <int dim, typename Number = double>
  class DiagonallyImplicitRungeKuttaIntegrator
  {
  public:
    DiagonallyImplicitRungeKuttaIntegrator(
      const unsigned int                        n_stages,
      const AdvectionOperation<dim, fe_degree> &op)
      : op(const_cast<AdvectionOperation<dim, fe_degree> &>(op))
      , n_accumulated_iterations(0)
      , n_solutions(0)
    {
      b.resize(n_stages);
      A.reinit(n_stages, n_stages);
      if (n_stages == 1)
        {
          // backward Euler
          b[0] = 1.0;
        }
      else if (n_stages == 3)
        {
          // third order, stiffly stable, by Alexander (or p 77/formula (229)
          // of Kennedy & Carpenter, 2016)
          const double gamma = 0.4358665215084589994160194;
          const double alpha = 1 + gamma * (-4 + 2 * gamma);
          const double beta  = -1 + gamma * (6 + gamma * (-9 + 3 * gamma));
          b                  = {{(-1 + 4 * gamma) / (4 * beta),
                                 -0.75 * alpha * alpha / beta,
                                 gamma}};
          for (unsigned int d = 0; d < 3; ++d)
            A(d, d) = gamma;
          const double c2 = (2 + gamma * (-9 + 6 * gamma)) / (3 * alpha);
          A(1, 0)         = c2 - gamma;
        }
      else if (n_stages == 4)
        {
          // third order, stiffly stable, by formula (237) on page 82 of
          // Kennedy & Carpenter, 2016
          const double gamma = 9. / 40.;
          b = {{4032. / 9943., 6929. / 15485., -723. / 9272., gamma}};
          for (unsigned int d = 0; d < 4; ++d)
            A(d, d) = gamma;
          A(1, 0) = 163. / 520.;
          A(2, 0) = -6481433. / 8838675.;
          A(2, 1) = 87795409. / 70709400.;
        }
      else if (n_stages == 7)
        {
          // ESDIRK4(3)7L[2]SA from Kennedy & Carpenter, Appl. Numer. Math.,
          // 2019, Table A.2. page 242
          const double gamma = 1. / 8.;

          b = {{-5649241495537. / 14093099002237.,
                -5649241495537. / 14093099002237.,
                5718691255176. / 6089204655961.,
                2199600963556. / 4241893152925.,
                8860614275765. / 11425531467341.,
                -3696041814078. / 6641566663007.,
                gamma}};

          A(0, 0) = 0.;
          for (unsigned int d = 1; d < n_stages; ++d)
            A(d, d) = gamma;

          A(2, 1) = -39188347878. / 1513744654945.;
          A(3, 1) = 1748874742213. / 5168247530883.;
          A(3, 2) = -1748874742213. / 5795261096931.;
          A(4, 1) = -6429340993097. / 17896796106705.;
          A(4, 2) = 9711656375562. / 10370074603625.;
          A(4, 3) = 1137589605079. / 3216875020685.;
          A(5, 1) = 405169606099. / 1734380148729.;
          A(5, 2) = -264468840649. / 6105657584947.;
          A(5, 3) = 118647369377. / 6233854714037.;
          A(5, 4) = 683008737625. / 4934655825458.;
          for (unsigned int d = 1; d < 6; ++d)
            A(d, 0) = A(d, 1);
        }
      else
        {
          AssertThrow(false,
                      ExcMessage("A scheme with " + std::to_string(n_stages) +
                                 " is not implemented!"));
        }
      for (unsigned int d = 0; d < n_stages; ++d)
        A(n_stages - 1, d) = b[d];

      c.resize(n_stages);
      for (unsigned int i = 0; i < n_stages; ++i)
        for (unsigned int j = 0; j <= i; ++j)
          c[i] += A(i, j);
    }

    void
    perform_time_step(LinearAlgebra::distributed::Vector<Number> &solution,
                      LinearAlgebra::distributed::Vector<Number> &tmp,
                      const double                                time,
                      const double time_step) const
    {
      std::vector<LinearAlgebra::distributed::Vector<Number>> ki(b.size());
      LinearAlgebra::distributed::Vector<Number>              tmp2;
      tmp2                                    = solution;
      const unsigned int n_stages             = b.size();
      unsigned int       first_implicit_stage = 0;
      const double       gamma                = A(n_stages - 1, n_stages - 1);

      BlockJacobi<AdvectionOperation<dim, fe_degree>> preconditioner(op);

      if (A(0, 0) == 0.0)
        {
          // explicit first stage
          first_implicit_stage = 1;
          ki[0].reinit(solution);
          op.set_time(time, 0.);
          op.vmult(ki[0], solution);
          op.apply_inverse_mass_matrix(ki[0]);

          solution.add(-b[0] * time_step, ki[0]);
          ki[0] *= gamma * time_step;
        }
      for (unsigned int stage = first_implicit_stage; stage < n_stages; ++stage)
        {
          ki[stage].reinit(solution);
          op.set_time(time + c[stage] * time_step, 0.);

          // note that we store what is called ki in Runge-Kutta methods as ki
          // / (gamma * dt) for simpler manipulation
          for (unsigned int r = 1; r < stage; ++r)
            tmp2.add((A(stage - 1, r - 1) - A(stage, r - 1)) / gamma,
                     ki[r - 1]);
          if (stage > 0)
            tmp2.add(-A(stage, stage - 1) / gamma, ki[stage - 1]);
          op.vmult(tmp, tmp2);
          op.set_time(time + c[stage] * time_step, gamma * time_step);

          SolverControl control(1000, tmp.l2_norm() * 1e-9);
          using SolverType =
            SolverFGMRES<LinearAlgebra::distributed::Vector<Number>>;
          typename SolverType::AdditionalData gmres_data;
          gmres_data.max_basis_size = 20;
          SolverType solver(control, gmres_data);
          solver.solve(op, ki[stage], tmp, preconditioner);
          n_accumulated_iterations += control.last_step();
          ++n_solutions;
          solution.add(-b[stage] / gamma, ki[stage]);
        }
    }

    std::pair<std::size_t, std::size_t>
    get_solver_statistics() const
    {
      return std::make_pair(n_accumulated_iterations, n_solutions);
    }

  private:
    AdvectionOperation<dim, fe_degree> &op;
    std::vector<double>                 b;
    std::vector<double>                 c;
    FullMatrix<double>                  A;

    mutable std::size_t n_accumulated_iterations;
    mutable std::size_t n_solutions;
  };



  template <int dim>
  class AdvectionProblem
  {
  public:
    typedef typename AdvectionOperation<dim, fe_degree>::Number Number;
    AdvectionProblem();
    void
    run(const unsigned int n_global_refinements);

  private:
    void
    make_grid(const unsigned int n_global_refinements);
    void
    setup_dofs();
    void
    output_results(const unsigned int timestep_number,
                   const Tensor<1, 3> mass_and_energy);

    LinearAlgebra::distributed::Vector<Number> solution;

    std::shared_ptr<Triangulation<dim>> triangulation;
    MappingQGeneric<dim>                mapping;
    FE_DGQArbitraryNodes<dim>           fe;
    DoFHandler<dim>                     dof_handler;

    IndexSet locally_relevant_dofs;

    double time, time_step;

    ConditionalOStream pcout;
  };



  template <int dim>
  AdvectionProblem<dim>::AdvectionProblem()
    : mapping(fe_degree)
    , fe(QGaussLobatto<1>(fe_degree + 1))
    , time(0)
    , time_step(0)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
#ifdef DEAL_II_WITH_P4EST
    if (dim > 1)
      triangulation =
        std::make_shared<parallel::distributed::Triangulation<dim>>(
          MPI_COMM_WORLD);
    else
#endif
      triangulation = std::make_shared<Triangulation<dim>>();
  }



  template <int dim>
  void
  AdvectionProblem<dim>::make_grid(const unsigned int n_global_refinements)
  {
    time      = 0;
    time_step = 0;
    triangulation->clear();
    Point<dim> p1;
    Point<dim> p2;
    for (unsigned int d = 0; d < dim; ++d)
      p2[d] = 1;
    std::vector<unsigned int> subdivisions(dim, 1);
    GridGenerator::subdivided_hyper_rectangle(*triangulation,
                                              subdivisions,
                                              p1,
                                              p2);

    if (periodic)
      {
        for (const auto &cell : triangulation->cell_iterators())
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->at_boundary(f))
              cell->face(f)->set_all_boundary_ids(f);
        std::vector<GridTools::PeriodicFacePair<
          typename Triangulation<dim>::cell_iterator>>
          periodic_faces;
        for (unsigned int d = 0; d < dim; ++d)
          GridTools::collect_periodic_faces(
            *triangulation, 2 * d, 2 * d + 1, d, periodic_faces);
        triangulation->add_periodicity(periodic_faces);
      }

    if (mesh_type == MeshType::deformed_cartesian)
      {
        DeformedCubeManifold<dim> manifold(0.0, 1.0, 0.12, 2);
        triangulation->set_all_manifold_ids(1);
        triangulation->set_manifold(1, manifold);

        std::vector<bool> vertex_touched(triangulation->n_vertices(), false);

        for (auto cell : triangulation->active_cell_iterators())
          {
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                 ++v)
              {
                if (vertex_touched[cell->vertex_index(v)] == false)
                  {
                    Point<dim> &vertex    = cell->vertex(v);
                    Point<dim>  new_point = manifold.push_forward(vertex);
                    vertex                = new_point;
                    vertex_touched[cell->vertex_index(v)] = true;
                  }
              }
          }
      }

    triangulation->refine_global(n_global_refinements);

    pcout << "   Number of elements:            "
          << triangulation->n_global_active_cells() << std::endl;
  }

  template <int dim>
  void
  AdvectionProblem<dim>::setup_dofs()
  {
#if DEAL_II_VERSION_GTE(9, 3, 0)
    dof_handler.reinit(*triangulation);
    dof_handler.distribute_dofs(fe);
#else
    dof_handler.initialize(*triangulation, fe);
#endif

    if (time == 0.)
      {
        pcout << "   Polynomial degree:             "
              << dof_handler.get_fe().degree << std::endl;
        pcout << "   Number of degrees of freedom:  " << dof_handler.n_dofs()
              << std::endl;
      }

    double min_vertex_distance = std::numeric_limits<double>::max();
    for (const auto &cell : triangulation->active_cell_iterators())
      min_vertex_distance =
        std::min(min_vertex_distance, cell->minimum_vertex_distance());
    const double glob_min_vertex_distance =
      Utilities::MPI::min(min_vertex_distance, MPI_COMM_WORLD);

    // Use hard-coded value for the maximal velocity of 2
    time_step = courant_number * glob_min_vertex_distance /
                TransportSpeed<dim>().value(Point<dim>()).norm();

    time_step = FINAL_TIME / std::ceil(FINAL_TIME / time_step);

    if (time == 0)
      pcout << "   Time step size: " << time_step
            << ", minimum vertex distance: " << glob_min_vertex_distance
            << std::endl
            << std::endl;
  }



  template <int dim>
  void
  AdvectionProblem<dim>::output_results(const unsigned int output_number,
                                        const Tensor<1, 3> mass_energy)
  {
    pcout << "   Time " << std::left << std::setw(6) << std::setprecision(3)
          << time << "  mass " << std::setprecision(10) << std::setw(16)
          << mass_energy[0] << "  energy " << std::setprecision(10)
          << std::setw(16) << mass_energy[1] << "  H1-semi "
          << std::setprecision(4) << std::setw(6) << mass_energy[2];

    if (!print_vtu)
      return;

    // Write output to a vtu file
    DataOut<dim>          data_out;
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(mapping,
                           fe_degree,
                           DataOut<dim>::curved_inner_cells);

    const std::string filename =
      "solution_" + Utilities::int_to_string(output_number, 3) + ".vtu";
    data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
  }



  template <int dim>
  void
  AdvectionProblem<dim>::run(const unsigned int n_global_refinements)
  {
    make_grid(n_global_refinements);
    setup_dofs();

    // Initialize the advection operator and the time integrator that will
    // perform all interesting steps
    AdvectionOperation<dim, fe_degree> advection_operator;
    advection_operator.reinit(dof_handler);
    advection_operator.initialize_dof_vector(solution);
    advection_operator.project_initial(solution);

    if (false && Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
      {
        DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
        DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
        SparsityPattern sp;
        sp.copy_from(dsp);
        SparseMatrix<Number> sp_mat(sp);
        advection_operator.compute_matrix(sp_mat);
        std::cout << "Matrix norm: " << sp_mat.frobenius_norm() << std::endl;
      }

    LinearAlgebra::distributed::Vector<Number> solution_copy = solution;
    LinearAlgebra::distributed::Vector<Number> rhs;
    rhs.reinit(solution);

    unsigned int n_output = 0;
    output_results(n_output++,
                   advection_operator.compute_mass_and_energy(solution));
    pcout << std::endl;

    Timer        timer;
    double       prep_time       = 0;
    double       sol_time        = 0;
    double       output_time     = 0;
    unsigned int timestep_number = 1;

    DiagonallyImplicitRungeKuttaIntegrator<dim> time_integrator(
      7, advection_operator);

    // This is the main time loop, asking the time integrator class to perform
    // the time step and update the content in the solution vector.
    while (time < FINAL_TIME - 1e-12)
      {
        timer.restart();

        time_integrator.perform_time_step(solution, rhs, time, time_step);

        time += time_step;
        timestep_number++;

        sol_time += timer.wall_time();

        timer.restart();

        if (static_cast<int>(time / output_tick) !=
              static_cast<int>((time - time_step) / output_tick) ||
            time >= FINAL_TIME - 1e-12)
          {
            output_results(
              n_output++, advection_operator.compute_mass_and_energy(solution));
            pcout << std::endl;
          }
        output_time += timer.wall_time();
      }

    solution_copy -= solution;
    pcout << std::endl
          << "   Distance |final solution - initial_condition|: "
          << solution_copy.linfty_norm() << std::endl;

    pcout << std::endl
          << "   Performed " << timestep_number << " time steps." << std::endl;

    pcout << "   Statistics of linear solver: n_systems = "
          << time_integrator.get_solver_statistics().second << ", avg_its = "
          << static_cast<double>(
               time_integrator.get_solver_statistics().first) /
               time_integrator.get_solver_statistics().second
          << std::endl;

    pcout << "   Average wall clock time per time step: "
          << (prep_time + sol_time) / timestep_number << "s, time per element: "
          << (prep_time + sol_time) / timestep_number /
               triangulation->n_global_active_cells()
          << "s" << std::endl;

    pcout << "   Spent " << output_time << "s on output, " << prep_time
          << "s on projection and " << sol_time << "s on solving." << std::endl;

    pcout << std::endl;

    // As 'advection_operator' goes out of scope, it will call its destructor
    // that prints the accumulated computing times over all time steps to
    // screen
  }
} // namespace DGAdvection



int
main(int argc, char **argv)
{
  using namespace DGAdvection;
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  // The only run-time parameter is to control the mesh size by the number the
  // initial mesh (consisting of a single line/square/cube) is refined by
  // doubling the number of elements for every increase in number. Thus, the
  // number of elements is given by 2^(dim * n_global_refinements)
  unsigned int n_global_refinements = 5;
  if (argc > 1)
    n_global_refinements = std::atoi(argv[1]);

  try
    {
      deallog.depth_console(0);

      // The actual dimension is selected by inserting the global constant
      // 'dimension' as the actual template argument here, rather than the
      // placeholder 'dim' used as *template* in the class definitions above.
      AdvectionProblem<dimension> advect_problem;
      advect_problem.run(n_global_refinements);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
