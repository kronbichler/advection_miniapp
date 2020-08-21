// This file is part of the advection_miniapp repository and subject to the
// LGPL license. See the LICENSE file in the top level directory of this
// project for details.

// Program for explicit time integration of the advection problem
// Author: Martin Kronbichler, Technical University of Munich, 2014-2020
//
// This program shares many similarities with the step-67 tutorial program of
// deal.II, see https://dealii.org/developer/doxygen/deal.II/step_67.html ,
// but it implements a simpler equation and is therefore ideal for learning
// about matrix-free evaluators.

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
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>

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
  // speed of a rotating vortex that is set here, only dimension 2 works
  const unsigned int dimension = 2;

  // The polynomial degree can be selected between 0 and any reasonable number
  // (around 30), depending on the dimension and the mesh size
  const unsigned int fe_degree = 4;

  // This parameter controls the mesh size by the number the initial mesh
  // (consisting of a single line/square/cube) is refined by doubling the
  // number of elements for every increase in number. Thus, the number of
  // elements is given by 2^(dim * n_global_refinements)
  const unsigned int n_min_global_refinements = 2;
  const unsigned int n_max_global_refinements = 3;

  // The time step size is controlled via this parameter as
  // dt = courant_number * min_h / (transport_norm * fe_degree^1.5)
  const double courant_number = 0.5;

  // 1: central flux, 0: classical upwind flux (= Lax-Friedrichs)
  const double flux_alpha = 0.0;

  // The final simulation time
  const double FINAL_TIME = 2.0;

  // Frequency of output
  const double output_tick = 0.1;

  enum LowStorageRungeKuttaScheme
  {
    stage_3_order_3, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_5_order_4, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_7_order_4, /* Tselios, Simos, 2007 */
    stage_9_order_5, /* Kennedy, Carpenter, Lewis, 2000 */
  };
  constexpr LowStorageRungeKuttaScheme lsrk_scheme = stage_5_order_4;

  // Whether to mesh the domain with Cartesian mesh elements or with curved
  // elements (more memory transfer -> slower)
  enum class MeshType
  {
    cartesian,
    deformed_cartesian,
    inscribed_circle
  };
  constexpr MeshType mesh_type = MeshType::cartesian;

  // Whether to set periodic boundary conditions on the domain (needs periodic
  // solution as well)
  const bool periodic = true;

  // Switch to change between a conservative formulation of the advection term
  // (factor 0) or a skew-symmetric one (factor 0.5)
  const double factor_skew = 0.0;

  // Switch to enable Gauss-Lobatto quadrature (true) or Gauss quadrature
  // (false)
  const bool use_gl_quad = false;

  // Switch to enable Gauss--Lobatto quadrature for the inverse mass
  // matrix. If false, use Gauss quadrature
  const bool use_gl_quad_mass = false;

  // Enable high-frequency components in the solution
  const bool high_frequency_sol = false;

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
      , wave_number(4.)
    {
      advection[0] = 1.1;
      if (dim > 1)
        advection[1] = 0.15;
      if (dim > 2)
        advection[2] = -0.05;
    }

    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const
    {
      return value<double>(p);
    }

    template <typename Number>
    Number
    value(const Point<dim, Number> &p) const
    {
      double t = this->get_time();
      Number result =
        std::sin(wave_number * (p[0] - t * advection[0]) * numbers::PI);
      for (unsigned int d = 1; d < dim; ++d)
        result *=
          std::cos(wave_number * (p[d] - t * advection[d]) * numbers::PI);
      if (high_frequency_sol)
        {
          Number noise =
            std::sin(16. * (p[0] - t * advection[0]) * numbers::PI);
          for (unsigned int d = 1; d < dim; ++d)
            noise *= std::sin(16. * (p[d] - t * advection[d]) * numbers::PI);
          result += 0.1 * noise;
          noise = std::sin(64. * (p[0] - t * advection[0]) * numbers::PI);
          for (unsigned int d = 1; d < dim; ++d)
            noise *= std::sin(64. * (p[d] - t * advection[d]) * numbers::PI);
          result += 0.0125 * noise;
        }
      return result;
    }

    Tensor<1, dim>
    get_transport_direction() const
    {
      return advection;
    }

  protected:
    Tensor<1, dim> advection;
    const double   wave_number;
  };



  // This class can be used to verify the spatial operator by applying the
  // operator on the ExactSolution field and comparing the result to this
  // class.
  template <int dim>
  class ExactSolutionTimeDer : public ExactSolution<dim>
  {
  public:
    ExactSolutionTimeDer(const double time = 0.)
      : ExactSolution<dim>(time)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const
    {
      Assert(high_frequency_sol == false, ExcNotImplemented());

      const double time = this->get_time();
      const double PI   = 3.14159265358979323846;
      double       position[dim];
      for (unsigned int d = 0; d < dim; ++d)
        position[d] = p[d] - time * this->advection[d];
      double result = -this->advection[0] * this->wave_number * PI *
                      std::cos(this->wave_number * position[0] * PI);
      for (unsigned int d = 1; d < dim; ++d)
        result *= std::cos(this->wave_number * position[d] * PI);
      double add =
        this->wave_number * PI * std::sin(this->wave_number * PI * position[0]);
      if (dim == 2)
        add *=
          this->advection[1] * std::sin(this->wave_number * PI * position[1]);
      else if (dim == 3)
        add *=
          (this->advection[1] * std::sin(this->wave_number * PI * position[1]) *
             std::cos(this->wave_number * position[2] * PI) +
           this->advection[2] * std::cos(this->wave_number * PI * position[1]) *
             std::sin(this->wave_number * position[2] * PI));
      return result + add;
    }
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
    push_forward(const Point<dim> &chart_point) const
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
    pull_back(const Point<dim> &space_point) const
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



  // Implementation of the advection operation
  template <int dim, int fe_degree>
  class AdvectionOperation
  {
  public:
    typedef double Number;

    AdvectionOperation()
      : computing_times(3)
    {}

    void
    reinit(const DoFHandler<dim> &dof_handler);

    void
    initialize_dof_vector(LinearAlgebra::distributed::Vector<Number> &vec)
    {
      data.initialize_dof_vector(vec);
    }

    ~AdvectionOperation()
    {
      if (computing_times[2] > 0)
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "Advection operator evaluated "
                      << (std::size_t)computing_times[2] << " times."
                      << std::endl
                      << "Time evaluate (min / avg / max): ";
          Utilities::MPI::MinMaxAvg data =
            Utilities::MPI::min_max_avg(computing_times[0], MPI_COMM_WORLD);
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << data.min << " (proc_" << data.min_index << ") / "
                      << data.avg << " / " << data.max << " (proc_"
                      << data.max_index << ")" << std::endl;
          data =
            Utilities::MPI::min_max_avg(computing_times[1], MPI_COMM_WORLD);
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "Time inv mass (min / avg / max): " << data.min
                      << " (proc_" << data.min_index << ") / " << data.avg
                      << " / " << data.max << " (proc_" << data.max_index << ")"
                      << std::endl;
        }
    }

    void
    apply(const LinearAlgebra::distributed::Vector<Number> &src,
          LinearAlgebra::distributed::Vector<Number> &      dst,
          const double                                      current_time);

    void
    perform_stage(const Number current_time,
                  const Number factor_solution,
                  const Number factor_ai,
                  const LinearAlgebra::distributed::Vector<Number> &current_ri,
                  LinearAlgebra::distributed::Vector<Number> &      vec_ki,
                  LinearAlgebra::distributed::Vector<Number> &      solution,
                  LinearAlgebra::distributed::Vector<Number> &next_ri) const;

    void
    project_initial(LinearAlgebra::distributed::Vector<Number> &dst) const;

    Tensor<1, 3>
    compute_mass_and_energy(
      const LinearAlgebra::distributed::Vector<Number> &vec) const;

  private:
    MatrixFree<dim, Number> data;
    mutable double          time;

    mutable std::vector<double> computing_times;

    void
    apply_mass_matrix(const LinearAlgebra::distributed::Vector<Number> &src,
                      LinearAlgebra::distributed::Vector<Number> &      dst);

    void
    local_apply_inverse_mass_matrix(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;

    void
    local_apply_domain(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;

    void
    local_apply_inner_face(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;
    void
    local_apply_boundary_face(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;
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
       update_values);
    additional_data.mapping_update_flags_boundary_faces =
      (update_JxW_values | update_normal_vectors | update_quadrature_points |
       update_values);

    AffineConstraints<double> dummy;
    dummy.close();
    data.reinit(mapping,
                {{&dof_handler}},
                std::vector<const AffineConstraints<double> *>{{&dummy}},
                std::vector<Quadrature<1>>{{quadrature, quadrature_mass}},
                additional_data);
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::local_apply_domain(
    const MatrixFree<dim, Number> &                   data,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval(data);

    ExactSolution<dim> solution(time);

    // convert convection velocity from a tensor of double-precision values
    // (Tensor<1,dim>) to a tensor of VectorizedArray<Number> values, i.e.,
    // some vectorized data types to use SIMD
    // (single-instruction/multiple-data), with the same entry on all
    // quadrature points
    Tensor<1, dim, VectorizedArray<Number>> speed;
    for (unsigned int d = 0; d < dim; ++d)
      speed[d] = solution.get_transport_direction()[d];

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);

        // compute u^h(x) from src
        eval.gather_evaluate(src, true, true);

        // loop over quadrature points and compute the local volume flux
        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            const auto u           = eval.get_value(q);
            const auto gradu       = eval.get_gradient(q);
            const auto volume_flux = (1.0 - factor_skew) * speed * u;
            eval.submit_gradient(volume_flux, q);
            eval.submit_value(-factor_skew * (speed * gradu), q);
          }

        // multiply by nabla v^h(x) and sum
        eval.integrate_scatter(true, true, dst);
      }
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::local_apply_inner_face(
    const MatrixFree<dim, Number> &                   data,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     face_range) const
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

    ExactSolution<dim>                      solution(time);
    Tensor<1, dim, VectorizedArray<Number>> speed;
    for (unsigned int d = 0; d < dim; ++d)
      speed[d] = solution.get_transport_direction()[d];

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_minus.gather_evaluate(src, true, false);
        eval_plus.reinit(face);
        eval_plus.gather_evaluate(src, true, false);

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const auto u_minus             = eval_minus.get_value(q);
            const auto u_plus              = eval_plus.get_value(q);
            const auto normal_vector_minus = eval_minus.get_normal_vector(q);

            const auto normal_times_speed = speed * normal_vector_minus;
            const auto flux_times_normal_of_minus =
              0.5 * ((u_minus + u_plus) * normal_times_speed +
                     (1.0 - flux_alpha) * std::abs(normal_times_speed) *
                       (u_minus - u_plus));

            // We want to test 'flux_times_normal' by the test function, which
            // is called 'FEEvaluation::submit_value(). We need a minus sign
            // for the minus side (interior face) because the boundary term is
            // located on the right hand side and should get a minus sign. On
            // the exterior/plus side, the normal vector has the opposite
            // sign. Instead of recomputing the flux times the normal vector
            // of the plus side, we simply switch the sign here
            eval_minus.submit_value(-flux_times_normal_of_minus +
                                      (factor_skew * normal_times_speed) *
                                        u_minus,
                                    q);
            eval_plus.submit_value(flux_times_normal_of_minus -
                                     (factor_skew * normal_times_speed) *
                                       u_plus,
                                   q);
          }

        eval_minus.integrate_scatter(true, false, dst);
        eval_plus.integrate_scatter(true, false, dst);
      }
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::local_apply_boundary_face(
    const MatrixFree<dim, Number> &                   data,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     face_range) const
  {
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_minus(data,
                                                                          true);

    ExactSolution<dim>                      solution(time);
    Tensor<1, dim, VectorizedArray<Number>> speed;
    for (unsigned int d = 0; d < dim; ++d)
      speed[d] = solution.get_transport_direction()[d];

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_minus.gather_evaluate(src, true, false);

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            // Dirichlet boundary
            const auto u_minus       = eval_minus.get_value(q);
            const auto normal_vector = eval_minus.get_normal_vector(q);

            // Compute the outer solution value
            const auto u_plus = solution.value(eval_minus.quadrature_point(q));

            // compute the flux
            const auto normal_times_speed = normal_vector * speed;
            const auto flux_times_normal =
              0.5 * ((u_minus + u_plus) * normal_times_speed +
                     (1.0 - flux_alpha) * std::abs(normal_times_speed) *
                       (u_minus - u_plus));

            // must use '-' sign because we move the advection terms to the
            // right hand side where we have a minus sign
            eval_minus.submit_value(-flux_times_normal + factor_skew *
                                                           normal_times_speed *
                                                           u_minus,
                                    q);
          }

        eval_minus.integrate_scatter(true, false, dst);
      }
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, Number> &                   data,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval(data, 0, 1);

    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, 1, Number>
      inverse(eval);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval.read_dof_values(src);

        inverse.apply(eval.begin_dof_values(), eval.begin_dof_values());

        eval.set_dof_values(dst);
      }
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::apply(
    const LinearAlgebra::distributed::Vector<Number> &src,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const double                                      current_time)
  {
    time = current_time;

    Timer timer;
    data.loop(&AdvectionOperation<dim, fe_degree>::local_apply_domain,
              &AdvectionOperation<dim, fe_degree>::local_apply_inner_face,
              &AdvectionOperation<dim, fe_degree>::local_apply_boundary_face,
              this,
              dst,
              src,
              true,
              MatrixFree<dim, Number>::DataAccessOnFaces::values,
              MatrixFree<dim, Number>::DataAccessOnFaces::values);
    computing_times[0] += timer.wall_time();

    timer.restart();
    data.cell_loop(
      &AdvectionOperation<dim, fe_degree>::local_apply_inverse_mass_matrix,
      this,
      dst,
      dst);
    computing_times[1] += timer.wall_time();

    computing_times[2] += 1.;
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::perform_stage(
    const Number                                      current_time,
    const Number                                      factor_solution,
    const Number                                      factor_ai,
    const LinearAlgebra::distributed::Vector<Number> &current_ri,
    LinearAlgebra::distributed::Vector<Number> &      vec_ki,
    LinearAlgebra::distributed::Vector<Number> &      solution,
    LinearAlgebra::distributed::Vector<Number> &      next_ri) const
  {
    time = current_time;

    Timer timer;

    data.loop(&AdvectionOperation<dim, fe_degree>::local_apply_domain,
              &AdvectionOperation<dim, fe_degree>::local_apply_inner_face,
              &AdvectionOperation<dim, fe_degree>::local_apply_boundary_face,
              this,
              vec_ki,
              current_ri,
              true,
              MatrixFree<dim, Number>::DataAccessOnFaces::values,
              MatrixFree<dim, Number>::DataAccessOnFaces::values);
    computing_times[0] += timer.wall_time();

    timer.restart();
    data.cell_loop(
      &AdvectionOperation<dim, fe_degree>::local_apply_inverse_mass_matrix,
      this,
      next_ri,
      vec_ki,
      std::function<void(const unsigned int, const unsigned int)>(),
      [&](const unsigned int start_range, const unsigned int end_range) {
        const Number ai = factor_ai;
        const Number bi = factor_solution;
        if (ai == Number())
          {
            DEAL_II_OPENMP_SIMD_PRAGMA
            for (unsigned int i = start_range; i < end_range; ++i)
              {
                const Number k_i          = next_ri.local_element(i);
                const Number sol_i        = solution.local_element(i);
                solution.local_element(i) = sol_i + bi * k_i;
              }
          }
        else
          {
            DEAL_II_OPENMP_SIMD_PRAGMA
            for (unsigned int i = start_range; i < end_range; ++i)
              {
                const Number k_i          = next_ri.local_element(i);
                const Number sol_i        = solution.local_element(i);
                solution.local_element(i) = sol_i + bi * k_i;
                next_ri.local_element(i)  = sol_i + ai * k_i;
              }
          }
      });
    computing_times[1] += timer.wall_time();

    computing_times[2] += 1.;
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
    dst.zero_out_ghosts();
    for (unsigned int cell = 0; cell < data.n_macro_cells(); ++cell)
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
        phi.gather_evaluate(vec, true, true);
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



  // Implementation of time integrators similarly to step-67 of deal.II
  class LowStorageRungeKuttaIntegrator
  {
  public:
    LowStorageRungeKuttaIntegrator(const LowStorageRungeKuttaScheme scheme)
    {
      switch (scheme)
        {
          case stage_3_order_3:
            {
              bi = {{0.245170287303492, 0.184896052186740, 0.569933660509768}};
              ai = {{0.755726351946097, 0.386954477304099}};

              break;
            }

          case stage_5_order_4:
            {
              bi = {{1153189308089. / 22510343858157.,
                     1772645290293. / 4653164025191.,
                     -1672844663538. / 4480602732383.,
                     2114624349019. / 3568978502595.,
                     5198255086312. / 14908931495163.}};
              ai = {{970286171893. / 4311952581923.,
                     6584761158862. / 12103376702013.,
                     2251764453980. / 15575788980749.,
                     26877169314380. / 34165994151039.}};

              break;
            }

          case stage_7_order_4:
            {
              bi = {{0.0941840925477795334,
                     0.149683694803496998,
                     0.285204742060440058,
                     -0.122201846148053668,
                     0.0605151571191401122,
                     0.345986987898399296,
                     0.186627171718797670}};
              ai = {{0.241566650129646868 + bi[0],
                     0.0423866513027719953 + bi[1],
                     0.215602732678803776 + bi[2],
                     0.232328007537583987 + bi[3],
                     0.256223412574146438 + bi[4],
                     0.0978694102142697230 + bi[5]}};

              break;
            }

          case stage_9_order_5:
            {
              bi = {{2274579626619. / 23610510767302.,
                     693987741272. / 12394497460941.,
                     -347131529483. / 15096185902911.,
                     1144057200723. / 32081666971178.,
                     1562491064753. / 11797114684756.,
                     13113619727965. / 44346030145118.,
                     393957816125. / 7825732611452.,
                     720647959663. / 6565743875477.,
                     3559252274877. / 14424734981077.}};
              ai = {{1107026461565. / 5417078080134.,
                     38141181049399. / 41724347789894.,
                     493273079041. / 11940823631197.,
                     1851571280403. / 6147804934346.,
                     11782306865191. / 62590030070788.,
                     9452544825720. / 13648368537481.,
                     4435885630781. / 26285702406235.,
                     2357909744247. / 11371140753790.}};

              break;
            }

          default:
            AssertThrow(false, ExcNotImplemented());
        }
    }

    unsigned int
    n_stages() const
    {
      return bi.size();
    }

    template <typename VectorType, typename Operator>
    void
    perform_time_step(const Operator &pde_operator,
                      const double    current_time,
                      const double    time_step,
                      VectorType &    solution,
                      VectorType &    vec_ri,
                      VectorType &    vec_ki) const
    {
      AssertDimension(ai.size() + 1, bi.size());

      pde_operator.perform_stage(current_time,
                                 bi[0] * time_step,
                                 ai[0] * time_step,
                                 solution,
                                 vec_ri,
                                 solution,
                                 vec_ri);
      double sum_previous_bi = 0;
      for (unsigned int stage = 1; stage < bi.size(); ++stage)
        {
          const double c_i = sum_previous_bi + ai[stage - 1];
          pde_operator.perform_stage(current_time + c_i * time_step,
                                     bi[stage] * time_step,
                                     (stage == bi.size() - 1 ?
                                        0 :
                                        ai[stage] * time_step),
                                     vec_ri,
                                     vec_ki,
                                     solution,
                                     vec_ri);
          sum_previous_bi += bi[stage - 1];
        }
    }

  private:
    std::vector<double> bi;
    std::vector<double> ai;
  };



  template <int dim>
  class AdvectionProblem
  {
  public:
    typedef typename AdvectionOperation<dim, fe_degree>::Number Number;
    AdvectionProblem();
    void
    run(const unsigned int n_refinements);

  private:
    void
    make_grid(const unsigned int n_refinements);
    void
    setup_dofs();
    void
    output_results(const unsigned int timestep_number,
                   const Tensor<1, 3> mass_and_energy);

    LinearAlgebra::distributed::Vector<Number> solution;

    std::shared_ptr<Triangulation<dim>> triangulation;
    MappingQGeneric<dim>                mapping;
    FE_DGQ<dim>                         fe;
    DoFHandler<dim>                     dof_handler;

    IndexSet locally_relevant_dofs;

    double time, time_step;

    ConditionalOStream pcout;
  };



  template <int dim>
  AdvectionProblem<dim>::AdvectionProblem()
    : mapping(fe_degree)
    , fe(fe_degree)
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
  AdvectionProblem<dim>::make_grid(const unsigned int n_refinements)
  {
    time      = 0;
    time_step = 0;
    triangulation->clear();
    Point<dim> p1;
    Point<dim> p2;
    for (unsigned int d = 0; d < dim; ++d)
      p2[d] = 1;
    std::vector<unsigned int> subdivisions(dim, 1);
    if (mesh_type == MeshType::inscribed_circle)
      {
        Triangulation<dim> tria1, tria2;
        Point<dim>         center;
        for (unsigned int d = 0; d < dim; ++d)
          center[d] = 0.5;
        if (dim == 3)
          GridGenerator::hyper_shell(
            tria1, center, 0.2, 0.5 * std::sqrt(dim), 2 * dim);
        else if (dim == 2)
          {
            GridGenerator::hyper_shell(
              tria1, Point<dim>(), 0.2, 0.5 * std::sqrt(dim), 2 * dim);
            GridTools::rotate(numbers::PI * 0.25, tria1);
            GridTools::shift(center, tria1);
          }
        else
          {
            AssertThrow(false, ExcNotImplemented());
          }
        GridGenerator::hyper_ball(tria2, center, 0.2);
        GridGenerator::merge_triangulations(tria1, tria2, *triangulation);
        triangulation->reset_all_manifolds();
        triangulation->set_all_manifold_ids(0);
        for (const auto &cell : triangulation->cell_iterators())
          {
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              {
                bool face_at_sphere_boundary = true;
                for (unsigned int v = 0;
                     v < GeometryInfo<dim - 1>::vertices_per_cell;
                     ++v)
                  if (std::abs(cell->face(f)->vertex(v).distance(center) -
                               0.2) > 1e-12)
                    face_at_sphere_boundary = false;
                if (face_at_sphere_boundary)
                  cell->face(f)->set_all_manifold_ids(1);
              }
          }
        const SphericalManifold<dim> spherical_manifold(center);
        triangulation->set_manifold(1, spherical_manifold);
        TransfiniteInterpolationManifold<dim> transfinite_manifold;
        transfinite_manifold.initialize(*triangulation);
        triangulation->set_manifold(0, transfinite_manifold);

        if (dim == 2 && periodic)
          {
            for (const auto &cell : triangulation->cell_iterators())
              for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell;
                   ++f)
                for (unsigned int d = 0; d < dim; ++d)
                  if (std::abs(cell->face(f)->center()[d]) < 1e-12)
                    cell->face(f)->set_all_boundary_ids(2 * d);
                  else if (std::abs(cell->face(f)->center()[d] - 1.) < 1e-12)
                    cell->face(f)->set_all_boundary_ids(2 * d + 1);
            std::vector<GridTools::PeriodicFacePair<
              typename Triangulation<dim>::cell_iterator>>
              periodic_faces;
            for (unsigned int d = 0; d < dim; ++d)
              GridTools::collect_periodic_faces(
                *triangulation, 2 * d, 2 * d + 1, d, periodic_faces);
            triangulation->add_periodicity(periodic_faces);
          }
      }
    else
      {
        GridGenerator::subdivided_hyper_rectangle(*triangulation,
                                                  subdivisions,
                                                  p1,
                                                  p2);

        if (periodic)
          {
            for (const auto &cell : triangulation->cell_iterators())
              for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell;
                   ++f)
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

            std::vector<bool> vertex_touched(triangulation->n_vertices(),
                                             false);

            for (auto cell : triangulation->active_cell_iterators())
              {
                for (unsigned int v = 0;
                     v < GeometryInfo<dim>::vertices_per_cell;
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
      }

    triangulation->refine_global(n_refinements);

    pcout << "   Number of elements:            "
          << triangulation->n_global_active_cells() << std::endl;
  }



  template <int dim>
  void
  AdvectionProblem<dim>::setup_dofs()
  {
    dof_handler.initialize(*triangulation, fe);

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

    ExactSolution<dim> exact;
    time_step = courant_number * glob_min_vertex_distance /
                exact.get_transport_direction().norm() /
                std::pow(static_cast<double>(std::max(1U, fe_degree)), 1.5);

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
    Vector<double> norm_per_cell(triangulation->n_active_cells());

    LinearAlgebra::distributed::Vector<Number> zero_vector;
    zero_vector.reinit(solution);
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      zero_vector,
                                      ExactSolution<dim>(time),
                                      norm_per_cell,
                                      QGauss<dim>(fe.degree + 1),
                                      VectorTools::L2_norm);
    double solution_mag =
      VectorTools::compute_global_error(*triangulation,
                                        norm_per_cell,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      ExactSolution<dim>(time),
                                      norm_per_cell,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::L2_norm);
    double error = VectorTools::compute_global_error(*triangulation,
                                                     norm_per_cell,
                                                     VectorTools::L2_norm);
    pcout << "   Time" << std::setw(8) << std::setprecision(3) << time
          << "  error " << std::setprecision(5) << std::setw(10)
          << error / solution_mag << "  mass " << std::setprecision(10)
          << std::setw(16) << mass_energy[0] << "  energy "
          << std::setprecision(10) << std::setw(16) << mass_energy[1]
          << "  H1-semi " << std::setprecision(4) << std::setw(9)
          << mass_energy[2] << std::endl;

    if (!print_vtu)
      return;

    // Write output to a vtu file
    DataOut<dim>          data_out;
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    LinearAlgebra::distributed::Vector<double> solution_double(
      dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
    VectorTools::interpolate(mapping,
                             dof_handler,
                             ExactSolution<dim>(time),
                             solution_double);
    data_out.add_data_vector(solution_double, "analytic_solution");
    data_out.build_patches(mapping,
                           fe_degree,
                           DataOut<dim>::curved_inner_cells);

    const std::string filename =
      "solution_" + Utilities::int_to_string(output_number, 3) + ".vtu";
    data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
  }



  template <int dim>
  void
  AdvectionProblem<dim>::run(const unsigned int n_refinements)
  {
    make_grid(n_refinements);
    setup_dofs();

    // Initialize the advection operator and the time integrator that will
    // perform all interesting steps
    AdvectionOperation<dim, fe_degree> advection_operator;
    advection_operator.reinit(dof_handler);
    advection_operator.initialize_dof_vector(solution);
    advection_operator.project_initial(solution);

    unsigned int n_output = 0;
    output_results(n_output++,
                   advection_operator.compute_mass_and_energy(solution));

    LinearAlgebra::distributed::Vector<Number> rk_register_1(solution),
      rk_register_2(solution);
    const LowStorageRungeKuttaIntegrator time_integrator(lsrk_scheme);

    Timer        timer;
    double       wtime           = 0;
    double       output_time     = 0;
    unsigned int timestep_number = 1;

    // This is the main time loop, asking the time integrator class to perform
    // the time step and update the content in the solution vector.
    while (time < FINAL_TIME - 1e-12)
      {
        timer.restart();

        time_integrator.perform_time_step(advection_operator,
                                          time,
                                          time_step,
                                          solution,
                                          rk_register_1,
                                          rk_register_2);
        time += time_step;
        timestep_number++;

        wtime += timer.wall_time();

        timer.restart();

        if (static_cast<int>(time / output_tick) !=
              static_cast<int>((time - time_step) / output_tick) ||
            time >= FINAL_TIME - 1e-12)
          {
            output_results(
              n_output++, advection_operator.compute_mass_and_energy(solution));
          }
        output_time += timer.wall_time();
      }

    pcout << std::endl
          << "   Performed " << timestep_number << " time steps." << std::endl;

    pcout << "   Average wall clock time per time step: "
          << wtime / timestep_number << "s, time per element: "
          << wtime / timestep_number / triangulation->n_global_active_cells()
          << "s" << std::endl;

    pcout << "   Spent " << output_time << "s on output and " << wtime
          << "s on computations." << std::endl;

    pcout << std::endl;

    // As 'advection_operator' goes out of scope, it will call its constructor
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

  try
    {
      deallog.depth_console(0);

      // The actual dimension is selected by inserting the global constant
      // 'dimension' as the actual template argument here, rather than the
      // placeholder 'dim' used as *template* in the class definitions above.
      AdvectionProblem<dimension> advect_problem;
      for (unsigned int r = n_min_global_refinements;
           r <= n_max_global_refinements;
           ++r)
        advect_problem.run(r);
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
