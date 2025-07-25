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
  const unsigned int fe_degree = 6;

  // The time step size is controlled via this parameter as
  // dt = courant_number * min_h / (transport_norm * fe_degree^1.5)
  const double courant_number = 2;

  // Number of stages in IRK method
  const unsigned int n_stages = 3;

  // 0: central flux, 1: classical upwind flux (= Lax-Friedrichs)
  const double flux_alpha = 1.0;

  // The final simulation time
  const double FINAL_TIME = 8;

  // Frequency of output
  const double output_tick = 0.1;

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
      return std::exp(-400. * ((p[0] - 0.5) * (p[0] - 0.5) +
                               (p[1] - 0.75) * (p[1] - 0.75))) +
             std::exp(-100. * ((p[0] - 0.5) * (p[0] - 0.5) +
                               (p[1] - 0.25) * (p[1] - 0.25)));
    }
  };



  template <int dim>
  class TransportSpeed
  {
  public:
    TransportSpeed(const double time)
      : time(time)
    {}

    template <typename Number>
    Tensor<1, dim, Number>
    value(const Point<dim, Number> &p) const
    {
      const double factor = std::cos(numbers::PI * time / FINAL_TIME) * 2.;
      Tensor<1, dim, Number> result;

      result[0] = factor * (std::sin(2 * numbers::PI * p[1]) *
                              std::sin(numbers::PI * p[0]) *
                              std::sin(numbers::PI * p[0]) +
                            0.2 * std::sin(20 * numbers::PI * (p[0] + 0.2)) *
                              std::cos(20 * numbers::PI * (p[1] + 0.3)));
      result[1] = -factor * (std::sin(2 * numbers::PI * p[0]) *
                               std::sin(numbers::PI * p[1]) *
                               std::sin(numbers::PI * p[1]) +
                             0.2 * std::cos(20 * numbers::PI * (p[0] + 0.2)) *
                               std::sin(20 * numbers::PI * (p[1] + 0.3)));

      return result;
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



  std::shared_ptr<const Utilities::MPI::Partitioner>
  create_partitioner_multiple(
    const std::shared_ptr<const Utilities::MPI::Partitioner>
                      &scalar_partitioner,
    const unsigned int multiplicity)
  {
    IndexSet owned(multiplicity * scalar_partitioner->size());
    owned.add_range(multiplicity * scalar_partitioner->local_range().first,
                    multiplicity * scalar_partitioner->local_range().second);
    IndexSet ghosted(owned.size());
    for (auto it = scalar_partitioner->ghost_indices().begin_intervals();
         it != scalar_partitioner->ghost_indices().end_intervals();
         ++it)
      ghosted.add_range(multiplicity * (*it->begin()),
                        multiplicity * (it->last() + 1));
    return std::make_shared<Utilities::MPI::Partitioner>(
      owned, ghosted, scalar_partitioner->get_mpi_communicator());
  }



  class IRK
  {
  public:
    IRK()
    {
      A.reinit(n_stages, n_stages);
      if (n_stages == 1)
        A(0, 0) = 1.;
      else if (n_stages == 2)
        {
          A(0, 0) = 5. / 12.;
          A(0, 1) = -1. / 12.;
          A(1, 0) = 3. / 4.;
          A(1, 1) = 1. / 4.;
        }
      else if (n_stages == 3)
        {
          A(0, 0) = 0.19681547722366044;
          A(0, 1) = -0.06553542585019838;
          A(0, 2) = 0.02377097434822015;
          A(1, 0) = 0.3944243147390873;
          A(1, 1) = 0.29207341166522843;
          A(1, 2) = -0.04154875212599792;
          A(2, 0) = 0.37640306270046725;
          A(2, 1) = 0.5124858261884216;
          A(2, 2) = 1. / 9.;
        }
      else if (n_stages == 4)
        {
          A(0, 0) = 0.11299954065227377;
          A(0, 1) = -0.04030922142462728;
          A(0, 2) = 0.025802391441157962;
          A(0, 3) = -0.009904667574594527;
          A(1, 0) = 0.2343839541547278;
          A(1, 1) = 0.20689257722452742;
          A(1, 2) = -0.047857142857142855;
          A(1, 3) = 0.016047419401474628;
          A(2, 0) = 0.21668177697189483;
          A(2, 1) = 0.4061232017705644;
          A(2, 2) = 0.18903654485049834;
          A(2, 3) = -0.02418210484824596;
          A(3, 0) = 0.22046204620462045;
          A(3, 1) = 0.3881932021466905;
          A(3, 2) = 0.3288439955106622;
          A(3, 3) = 1. / 16.;
        }
      else if (n_stages == 5)
        {
          A(0, 0) = 0.072998864317903;
          A(0, 1) = -0.026735331107945;
          A(0, 2) = 0.018676929763984;
          A(0, 3) = -0.012879106093306;
          A(0, 4) = 0.005042839233882;
          A(1, 0) = 0.153775231479182;
          A(1, 1) = 0.146214867847493;
          A(1, 2) = -0.036444568905128;
          A(1, 3) = 0.021233063119304;
          A(1, 4) = -0.007935579902729;
          A(2, 0) = 0.140063045684809;
          A(2, 1) = 0.298967129491283;
          A(2, 2) = 0.167585070135249;
          A(2, 3) = -0.033969101686618;
          A(2, 4) = 0.010944288744193;
          A(3, 0) = 0.144894308109531;
          A(3, 1) = 0.276500068760161;
          A(3, 2) = 0.325797922910423;
          A(3, 3) = 0.128756753254908;
          A(3, 4) = -0.015708917378805;
          A(4, 0) = 0.143713560791217;
          A(4, 1) = 0.281356015149465;
          A(4, 2) = 0.311826522975736;
          A(4, 3) = 0.223103901083572;
          A(4, 4) = 1. / 25.;
        }
      b.reinit(A.m());
      for (unsigned int i = 0; i < b.size(); ++i)
        b(i) = A(A.m() - 1, i);
      c.reinit(A.m());
      for (unsigned int i = 0; i < c.size(); ++i)
        for (unsigned int j = 0; j < c.size(); ++j)
          c(i) += A(i, j);

      inv_A = A;
      inv_A.gauss_jordan();

      LAPACKFullMatrix<double> A_lap;
      A_lap.copy_from(inv_A);
      A_lap.compute_eigenvalues(true, false);
      eigenvalues.resize(A.n());
      for (unsigned int i = 0; i < A.n(); ++i)
        eigenvalues[i] = A_lap.eigenvalue(i);

      T     = A_lap.get_right_eigenvectors();
      inv_T = T;
      inv_T.gauss_jordan();
    }

    FullMatrix<double>                A, inv_A;
    Vector<double>                    b;
    Vector<double>                    c;
    std::vector<std::complex<double>> eigenvalues;
    FullMatrix<std::complex<double>>  T, inv_T;
  };



  template <int dim, int fe_degree, int n_stages, typename Number = double>
  class CellwiseOperator
  {
  public:
    CellwiseOperator(
      const Tensor<2, dim, VectorizedArray<Number>>                    &jac,
      const internal::MatrixFreeFunctions::UnivariateShapeData<Number> &shape,
      const Tensor<1, dim, VectorizedArray<Number>> *speed_cells,
      const Table<2, VectorizedArray<Number>>       &normal_speed_faces,
      const Quadrature<dim>                         &cell_quadrature,
      const Quadrature<dim - 1>                     &face_quadrature,
      const double                                   inv_dt,
      const FullMatrix<double>                      &inv_A,
      const std::array<Number, n_stages>            &time_factor)
      : jac(jac)
      , shape(shape)
      , speed_cells(speed_cells)
      , normal_speed_faces(normal_speed_faces)
      , cell_quadrature(cell_quadrature)
      , face_quadrature(face_quadrature)
      , inv_dt(inv_dt)
      , inv_A(inv_A)
      , time_factors(time_factor)
    {}

    void
    vmult(Vector<Number> &dst, const Vector<Number> &src) const
    {
      const VectorizedArray<Number> *src_ptr =
        reinterpret_cast<const VectorizedArray<Number> *>(src.data());
      VectorizedArray<Number> *dst_ptr =
        reinterpret_cast<VectorizedArray<Number> *>(dst.data());
      const unsigned int dofs_per_component =
        Utilities::pow(fe_degree + 1, dim);
      VectorizedArray<Number> gradients[n_stages][dim][dofs_per_component];

      // face integrals relevant to present cell
      for (unsigned int d = 0; d < dim; ++d)
        {
          const unsigned int stride  = Utilities::pow(fe_degree + 1, d);
          const unsigned int offset0 = d > 0 ? 1 : fe_degree + 1;
          const unsigned int offset1 =
            (dim > 2 ?
               (d == 2 ? (fe_degree + 1) : Utilities::pow(fe_degree + 1, 2)) :
               1);
          VectorizedArray<Number> surface_JxW = 1.;
          for (unsigned int e = 0; e < dim; ++e)
            if (d != e)
              surface_JxW *= jac[e][e];
          surface_JxW = 1.0 / surface_JxW;

          for (unsigned int i1 = 0; i1 < (dim > 2 ? fe_degree + 1 : 1); ++i1)
            for (unsigned int i0 = 0; i0 < (dim > 1 ? fe_degree + 1 : 1); ++i0)
              {
                VectorizedArray<Number> sum_l[n_stages] = {},
                                        sum_r[n_stages] = {};
                const auto *my_ptr = src_ptr + offset1 * i1 + offset0 * i0;
                for (unsigned int j = 0; j < fe_degree + 1;
                     ++j, my_ptr += stride)
                  {
                    for (unsigned int stage = 0; stage < n_stages; ++stage)
                      {
                        const VectorizedArray<Number> value =
                          my_ptr[stage * dofs_per_component];
                        sum_l[stage] +=
                          shape.quadrature_data_on_face[0][j] * value;
                        sum_r[stage] +=
                          shape.quadrature_data_on_face[0][fe_degree - j] *
                          value;
                      }
                  }

                const unsigned int q = (dim == 3 && d == 1) ?
                                         i0 * (fe_degree + 1) + i1 :
                                         i1 * (fe_degree + 1) + i0;
                {
                  for (unsigned int stage = 0; stage < n_stages; ++stage)
                    {
                      const auto speed =
                        normal_speed_faces(2 * d, q) * time_factors[stage];
                      const auto coefficient =
                        0.5 * (speed + flux_alpha * std::abs(speed)) -
                        factor_skew * speed;
                      sum_l[stage] = sum_l[stage] * coefficient *
                                     (face_quadrature.weight(q) * surface_JxW);
                    }
                }
                {
                  for (unsigned int stage = 0; stage < n_stages; ++stage)
                    {
                      const auto speed =
                        normal_speed_faces(2 * d + 1, q) * time_factors[stage];
                      const auto coefficient =
                        0.5 * (speed + flux_alpha * std::abs(speed)) -
                        factor_skew * speed;
                      sum_r[stage] = sum_r[stage] * coefficient *
                                     (face_quadrature.weight(q) * surface_JxW);
                    }
                }

                auto out_ptr = dst_ptr + offset1 * i1 + offset0 * i0;
                if (d == 0)
                  for (unsigned int j = 0; j < fe_degree + 1;
                       ++j, out_ptr += stride)
                    for (unsigned int stage = 0; stage < n_stages; ++stage)
                      out_ptr[stage * dofs_per_component] =
                        shape.quadrature_data_on_face[0][j] * sum_l[stage] +
                        shape.quadrature_data_on_face[0][fe_degree - j] *
                          sum_r[stage];
                else
                  for (unsigned int j = 0; j < fe_degree + 1;
                       ++j, out_ptr += stride)
                    for (unsigned int stage = 0; stage < n_stages; ++stage)
                      out_ptr[stage * dofs_per_component] +=
                        shape.quadrature_data_on_face[0][j] * sum_l[stage] +
                        shape.quadrature_data_on_face[0][fe_degree - j] *
                          sum_r[stage];
              }
        }

      internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                       dim,
                                       fe_degree + 1,
                                       fe_degree + 1,
                                       VectorizedArray<Number>,
                                       Number>
        eval(shape.shape_values_eo, shape.shape_gradients_collocation_eo, {});

      // volume integrals
      for (unsigned int stage = 0; stage < n_stages; ++stage)
        {
          eval.template gradients<0, true, false>(src_ptr +
                                                    stage * dofs_per_component,
                                                  gradients[stage][0]);
          if (dim > 1)
            eval.template gradients<1, true, false>(
              src_ptr + stage * dofs_per_component, gradients[stage][1]);
          if (dim > 2)
            eval.template gradients<2, true, false>(
              src_ptr + stage * dofs_per_component, gradients[stage][2]);
        }
      const VectorizedArray<Number> JxW = 1. / determinant(jac);
      for (unsigned int q = 0; q < dofs_per_component; ++q)
        {
          const auto              speed = speed_cells[q];
          VectorizedArray<Number> u[n_stages];
          VectorizedArray<Number> flux[n_stages];
          for (unsigned int stage = 0; stage < n_stages; ++stage)
            {
              u[stage] = src_ptr[q + stage * dofs_per_component];
              VectorizedArray<Number> speed_gradu =
                speed[0] * gradients[stage][0][q] * jac[0][0];
              for (unsigned int d = 1; d < dim; ++d)
                speed_gradu += speed[d] * gradients[stage][d][q] * jac[d][d];
              flux[stage] = (factor_skew * time_factors[stage]) * speed_gradu;
              const VectorizedArray<Number> result =
                (-1.0 + factor_skew) * time_factors[stage] * u[stage] *
                (JxW * cell_quadrature.weight(q));
              for (unsigned int d = 0; d < dim; ++d)
                gradients[stage][d][q] = result * speed[d] * jac[d][d];
            }
          // mass matrix part
          for (unsigned int stage = 0; stage < n_stages; ++stage)
            u[stage] *= VectorizedArray<Number>(inv_dt);
          for (unsigned int stage = 0; stage < n_stages; ++stage)
            {
              for (unsigned int s = 0; s < n_stages; ++s)
                flux[stage] += inv_A(stage, s) * u[s];
              dst_ptr[q + stage * dofs_per_component] +=
                flux[stage] * (JxW * cell_quadrature.weight(q));
            }
        }
      for (unsigned int stage = 0; stage < n_stages; ++stage)
        {
          eval.template gradients<0, false, true>(gradients[stage][0],
                                                  dst_ptr +
                                                    stage * dofs_per_component);
          if (dim > 1)
            eval.template gradients<1, false, true>(
              gradients[stage][1], dst_ptr + stage * dofs_per_component);
          if (dim > 2)
            eval.template gradients<2, false, true>(
              gradients[stage][2], dst_ptr + stage * dofs_per_component);
        }
    }

    void
    transform_to_collocation(const VectorizedArray<Number> *src_ptr,
                             Vector<Number>                &dst) const
    {
      internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                       dim,
                                       fe_degree + 1,
                                       fe_degree + 1,
                                       VectorizedArray<Number>,
                                       Number>
                               evaluator({}, {}, shape.inverse_shape_values_eo);
      VectorizedArray<Number> *dst_ptr =
        reinterpret_cast<VectorizedArray<Number> *>(dst.data());
      const unsigned int dofs_per_component =
        Utilities::pow(fe_degree + 1, dim);

      for (unsigned int d = 0; d < n_stages; ++d)
        {
          const VectorizedArray<Number> *in  = src_ptr + d * dofs_per_component;
          VectorizedArray<Number>       *out = dst_ptr + d * dofs_per_component;
          // Need to select 'apply' method with hessian slot because values
          // assume symmetries that do not exist in the inverse shapes
          evaluator.template hessians<0, true, false>(in, out);
          if (dim > 1)
            evaluator.template hessians<1, true, false>(out, out);
          if (dim > 2)
            evaluator.template hessians<2, true, false>(out, out);
        }
    }

    void
    transform_from_collocation(const Vector<Number>    &src,
                               VectorizedArray<Number> *dst_ptr) const
    {
      internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                       dim,
                                       fe_degree + 1,
                                       fe_degree + 1,
                                       VectorizedArray<Number>,
                                       Number>
        evaluator({}, {}, shape.inverse_shape_values_eo);
      const VectorizedArray<Number> *src_ptr =
        reinterpret_cast<const VectorizedArray<Number> *>(src.data());
      const unsigned int dofs_per_component =
        Utilities::pow(fe_degree + 1, dim);

      for (unsigned int d = 0; d < n_stages; ++d)
        {
          const VectorizedArray<Number> *in  = src_ptr + d * dofs_per_component;
          VectorizedArray<Number>       *out = dst_ptr + d * dofs_per_component;
          // Need to select 'apply' method with hessian slot because values
          // assume symmetries that do not exist in the inverse shapes
          evaluator.template hessians<0, false, false>(in, out);
          if (dim > 1)
            evaluator.template hessians<1, false, false>(out, out);
          if (dim > 2)
            evaluator.template hessians<2, false, false>(out, out);
        }
    }

  private:
    const Tensor<2, dim, VectorizedArray<Number>>                     jac;
    const internal::MatrixFreeFunctions::UnivariateShapeData<Number> &shape;
    const Tensor<1, dim, VectorizedArray<Number>> *speed_cells;
    const Table<2, VectorizedArray<Number>>       &normal_speed_faces;
    const Quadrature<dim>                         &cell_quadrature;
    const Quadrature<dim - 1>                     &face_quadrature;
    const Number                                   inv_dt;
    const FullMatrix<double>                      &inv_A;
    const std::array<Number, n_stages>             time_factors;
  };



  template <int dim, int fe_degree, int n_stages, typename Number = double>
  class CellwiseRungeKutta
  {
  public:
    CellwiseRungeKutta(
      const Tensor<2, dim, VectorizedArray<Number>> &jac,
      const internal::MatrixFreeFunctions::UnivariateShapeData<
        VectorizedArray<Number>>                    &shape,
      const Tensor<1, dim, VectorizedArray<Number>> *speed_cells,
      const Table<2, VectorizedArray<Number>>       &normal_speed_faces,
      const Quadrature<dim>                         &cell_quadrature,
      const Quadrature<dim - 1>                     &face_quadrature,
      const double                                   dt,
      const FullMatrix<double>                      &A,
      const std::array<Number, n_stages>            &time_factor)
      : jac(jac)
      , shape(shape)
      , speed_cells(speed_cells)
      , normal_speed_faces(normal_speed_faces)
      , cell_quadrature(cell_quadrature)
      , face_quadrature(face_quadrature)
      , dt(dt)
      , A(A)
      , time_factors(time_factor)
    {}

    void
    propagate(VectorizedArray<Number> *values) const
    {
      std::array<Number, 3> bi{
        {0.245170287303492, 0.184896052186740, 0.569933660509768}};
      std::array<Number, 2> ai{{0.755726351946097, 0.386954477304099}};
      const unsigned int n_dofs = Utilities::pow(fe_degree + 1, dim) * n_stages;
      VectorizedArray<Number> sol_tmp[n_dofs];
      VectorizedArray<Number> ki[n_dofs];

      for (unsigned int i = 0; i < n_dofs; ++i)
        values[i] = values[i] * dt;
      apply_inverse_mass(values);

      VectorizedArray<Number> max_dt = 0.;
      for (unsigned int q = 0; q < cell_quadrature.size(); ++q)
        max_dt = std::max((jac * speed_cells[q]).norm_square(), max_dt);
      max_dt *= std::max(std::abs(time_factors[0]),
                         std::abs(time_factors[n_stages - 1]));
      for (unsigned int v = 1; v < VectorizedArray<Number>::size(); ++v)
        max_dt[0] = std::max(max_dt[0], max_dt[v]);
      Number my_dt = std::min(
        dt, 0.05 / std::sqrt(Utilities::pow(fe_degree + 1, 3) * max_dt[0]));
      const unsigned int n_steps =
        static_cast<unsigned int>(std::round(dt / my_dt));
      my_dt = dt / n_steps;
      // std::cout << my_dt << " " << n_steps << "  ";

      for (unsigned int step = 0; step < n_steps; ++step)
        {
          apply(values, ki);
          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              sol_tmp[i] = values[i] + (my_dt * ai[0]) * ki[i];
              values[i] += (my_dt * bi[0]) * ki[i];
            }
          apply(sol_tmp, ki);
          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              sol_tmp[i] = values[i] + (my_dt * ai[1]) * ki[i];
              values[i] += (my_dt * bi[1]) * ki[i];
            }
          apply(sol_tmp, ki);
          for (unsigned int i = 0; i < n_dofs; ++i)
            values[i] += (my_dt * bi[2]) * ki[i];
        }
    }

    void
    apply(const VectorizedArray<Number> *src_ptr,
          VectorizedArray<Number>       *dst_ptr) const
    {
      constexpr unsigned int dofs_per_component =
        Utilities::pow(fe_degree + 1, dim);
      VectorizedArray<Number> gradients[dim][dofs_per_component];

      internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                       dim,
                                       fe_degree + 1,
                                       fe_degree + 1,
                                       VectorizedArray<Number>,
                                       Number>
        eval(shape.shape_values_eo, shape.shape_gradients_collocation_eo, {});

      // face integrals relevant to present cell
      for (unsigned int d = 0; d < dim; ++d)
        {
          const unsigned int stride  = Utilities::pow(fe_degree + 1, d);
          const unsigned int offset0 = d > 0 ? 1 : fe_degree + 1;
          const unsigned int offset1 =
            (dim > 2 ?
               (d == 2 ? (fe_degree + 1) : Utilities::pow(fe_degree + 1, 2)) :
               1);
          VectorizedArray<Number> surface_JxW = 1.;
          for (unsigned int e = 0; e < dim; ++e)
            if (d != e)
              surface_JxW *= jac[e][e];
          surface_JxW = 1.0 / surface_JxW;

          for (unsigned int i1 = 0; i1 < (dim > 2 ? fe_degree + 1 : 1); ++i1)
            for (unsigned int i0 = 0; i0 < (dim > 1 ? fe_degree + 1 : 1); ++i0)
              {
                VectorizedArray<Number> sum_l[n_stages] = {},
                                        sum_r[n_stages] = {};
                const auto *my_ptr = src_ptr + offset1 * i1 + offset0 * i0;
                for (unsigned int j = 0; j < fe_degree + 1;
                     ++j, my_ptr += stride)
                  {
                    for (unsigned int stage = 0; stage < n_stages; ++stage)
                      {
                        const VectorizedArray<Number> value =
                          my_ptr[stage * dofs_per_component];
                        sum_l[stage] +=
                          shape.quadrature_data_on_face[0][j] * value;
                        sum_r[stage] +=
                          shape.quadrature_data_on_face[0][fe_degree - j] *
                          value;
                      }
                  }

                const unsigned int q = (dim == 3 && d == 1) ?
                                         i0 * (fe_degree + 1) + i1 :
                                         i1 * (fe_degree + 1) + i0;
                {
                  for (unsigned int stage = 0; stage < n_stages; ++stage)
                    {
                      const auto speed =
                        normal_speed_faces(2 * d, q) * time_factors[stage];
                      const auto coefficient =
                        0.5 * (speed + flux_alpha * std::abs(speed)) -
                        factor_skew * speed;
                      sum_l[stage] = sum_l[stage] * coefficient *
                                     (face_quadrature.weight(q) * surface_JxW);
                    }
                }
                {
                  for (unsigned int stage = 0; stage < n_stages; ++stage)
                    {
                      const auto speed =
                        normal_speed_faces(2 * d + 1, q) * time_factors[stage];
                      const auto coefficient =
                        0.5 * (speed + flux_alpha * std::abs(speed)) -
                        factor_skew * speed;
                      sum_r[stage] = sum_r[stage] * coefficient *
                                     (face_quadrature.weight(q) * surface_JxW);
                    }
                }

                auto out_ptr = dst_ptr + offset1 * i1 + offset0 * i0;
                if (d == 0)
                  for (unsigned int j = 0; j < fe_degree + 1;
                       ++j, out_ptr += stride)
                    for (unsigned int stage = 0; stage < n_stages; ++stage)
                      out_ptr[stage * dofs_per_component] =
                        shape.quadrature_data_on_face[0][j] * sum_l[stage] +
                        shape.quadrature_data_on_face[0][fe_degree - j] *
                          sum_r[stage];
                else
                  for (unsigned int j = 0; j < fe_degree + 1;
                       ++j, out_ptr += stride)
                    for (unsigned int stage = 0; stage < n_stages; ++stage)
                      out_ptr[stage * dofs_per_component] +=
                        shape.quadrature_data_on_face[0][j] * sum_l[stage] +
                        shape.quadrature_data_on_face[0][fe_degree - j] *
                          sum_r[stage];
              }
        }

      const VectorizedArray<Number> JxW = 1. / determinant(jac);
      // volume integrals
      for (unsigned int stage = 0; stage < n_stages; ++stage)
        {
          eval.template gradients<0, true, false>(src_ptr +
                                                    stage * dofs_per_component,
                                                  gradients[0]);
          if (dim > 1)
            eval.template gradients<1, true, false>(
              src_ptr + stage * dofs_per_component, gradients[1]);
          if (dim > 2)
            eval.template gradients<2, true, false>(
              src_ptr + stage * dofs_per_component, gradients[2]);

          for (unsigned int q = 0; q < dofs_per_component; ++q)
            {
              const auto speed = speed_cells[q];
              const auto u     = src_ptr[q + stage * dofs_per_component];
              VectorizedArray<Number> speed_gradu =
                speed[0] * gradients[0][q] * jac[0][0];
              for (unsigned int d = 1; d < dim; ++d)
                speed_gradu += speed[d] * gradients[d][q] * jac[d][d];
              dst_ptr[q + stage * dofs_per_component] +=
                (factor_skew * time_factors[stage]) * speed_gradu *
                (JxW * cell_quadrature.weight(q));
              const VectorizedArray<Number> result =
                (-1.0 + factor_skew) * time_factors[stage] * u *
                (JxW * cell_quadrature.weight(q));
              for (unsigned int d = 0; d < dim; ++d)
                gradients[d][q] = result * speed[d] * jac[d][d];
            }

          eval.template gradients<0, false, true>(gradients[0],
                                                  dst_ptr +
                                                    stage * dofs_per_component);
          if (dim > 1)
            eval.template gradients<1, false, true>(
              gradients[1], dst_ptr + stage * dofs_per_component);
          if (dim > 2)
            eval.template gradients<2, false, true>(
              gradients[2], dst_ptr + stage * dofs_per_component);
        }

      for (unsigned int i = 0; i < n_stages * dofs_per_component; ++i)
        dst_ptr[i] = -dst_ptr[i];

      apply_inverse_mass(dst_ptr);
    }

    void
    apply_inverse_mass(VectorizedArray<Number> *dst_ptr) const
    {
      constexpr unsigned int dofs_per_component =
        Utilities::pow(fe_degree + 1, dim);
      const VectorizedArray<Number> inverse_jacobian_determinant =
        determinant(jac);
      VectorizedArray<Number> myA[n_stages][n_stages];
      for (unsigned int i = 0; i < n_stages; ++i)
        for (unsigned int j = 0; j < n_stages; ++j)
          myA[i][j] = A(i, j);

      for (unsigned int q = 0; q < dofs_per_component; ++q)
        {
          const VectorizedArray<Number> factor =
            (1. / cell_quadrature.weight(q)) * inverse_jacobian_determinant;
          VectorizedArray<Number> rhs[n_stages];
          for (unsigned int stage = 0; stage < n_stages; ++stage)
            rhs[stage] = dst_ptr[q + stage * dofs_per_component];
          for (unsigned int stage = 0; stage < n_stages; ++stage)
            {
              VectorizedArray<Number> sum = myA[stage][0] * rhs[0];
              for (unsigned int s = 1; s < n_stages; ++s)
                sum += myA[stage][s] * rhs[s];
              dst_ptr[q + stage * dofs_per_component] = sum * factor;
            }
        }
    }

    void
    transform_to_collocation(const VectorizedArray<Number> *src_ptr,
                             VectorizedArray<Number>       *dst_ptr) const
    {
      internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                       dim,
                                       fe_degree + 1,
                                       fe_degree + 1,
                                       VectorizedArray<Number>,
                                       Number>
                         evaluator({}, {}, shape.inverse_shape_values_eo);
      const unsigned int dofs_per_component =
        Utilities::pow(fe_degree + 1, dim);

      for (unsigned int d = 0; d < n_stages; ++d)
        {
          const VectorizedArray<Number> *in  = src_ptr + d * dofs_per_component;
          VectorizedArray<Number>       *out = dst_ptr + d * dofs_per_component;
          // Need to select 'apply' method with hessian slot because values
          // assume symmetries that do not exist in the inverse shapes
          evaluator.template hessians<0, true, false>(in, out);
          if (dim > 1)
            evaluator.template hessians<1, true, false>(out, out);
          if (dim > 2)
            evaluator.template hessians<2, true, false>(out, out);
        }
    }

    void
    transform_from_collocation(const VectorizedArray<Number> *src_ptr,
                               VectorizedArray<Number>       *dst_ptr) const
    {
      internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                       dim,
                                       fe_degree + 1,
                                       fe_degree + 1,
                                       VectorizedArray<Number>,
                                       Number>
                         evaluator({}, {}, shape.inverse_shape_values_eo);
      const unsigned int dofs_per_component =
        Utilities::pow(fe_degree + 1, dim);

      for (unsigned int d = 0; d < n_stages; ++d)
        {
          const VectorizedArray<Number> *in  = src_ptr + d * dofs_per_component;
          VectorizedArray<Number>       *out = dst_ptr + d * dofs_per_component;
          // Need to select 'apply' method with hessian slot because values
          // assume symmetries that do not exist in the inverse shapes
          evaluator.template hessians<0, false, false>(in, out);
          if (dim > 1)
            evaluator.template hessians<1, false, false>(out, out);
          if (dim > 2)
            evaluator.template hessians<2, false, false>(out, out);
        }
    }

  private:
    const Tensor<2, dim, VectorizedArray<Number>> jac;
    const internal::MatrixFreeFunctions::UnivariateShapeData<
      VectorizedArray<Number>>                    &shape;
    const Tensor<1, dim, VectorizedArray<Number>> *speed_cells;
    const Table<2, VectorizedArray<Number>>       &normal_speed_faces;
    const Quadrature<dim>                         &cell_quadrature;
    const Quadrature<dim - 1>                     &face_quadrature;
    const Number                                   dt;
    const FullMatrix<double>                      &A;
    const std::array<Number, n_stages>             time_factors;
  };



  template <int n_stages, typename Number = double>
  class CellwisePreconditioner
  {
  public:
    template <int dim>
    CellwisePreconditioner(const Quadrature<dim>    &cell_quadrature,
                           const FullMatrix<double> &A)
      : A(A)
    {
      inverse_quadrature_weight.resize(cell_quadrature.size());
      for (unsigned int q = 0; q < cell_quadrature.size(); ++q)
        inverse_quadrature_weight[q] = 1. / cell_quadrature.weight(q);
    }

    template <int dim>
    void
    reinit(const Tensor<2, dim, VectorizedArray<Number>> &jac)
    {
      inverse_jacobian_determinant = determinant(jac);
    }

    void
    vmult(Vector<Number> &dst, const Vector<Number> &src) const
    {
      const unsigned int dofs_per_component = inverse_quadrature_weight.size();
      const VectorizedArray<Number> *src_ptr =
        reinterpret_cast<const VectorizedArray<Number> *>(src.data());
      VectorizedArray<Number> *dst_ptr =
        reinterpret_cast<VectorizedArray<Number> *>(dst.data());
      constexpr unsigned int n_lanes = VectorizedArray<Number>::size();
      (void)n_lanes;
      AssertDimension(n_stages * n_lanes * dofs_per_component, dst.size());
      AssertDimension(n_stages * n_lanes * dofs_per_component, src.size());
      VectorizedArray<Number> myA[n_stages][n_stages];
      for (unsigned int i = 0; i < n_stages; ++i)
        for (unsigned int j = 0; j < n_stages; ++j)
          myA[i][j] = A(i, j);

      for (unsigned int q = 0; q < dofs_per_component; ++q)
        {
          const VectorizedArray<Number> factor =
            inverse_quadrature_weight[q] * inverse_jacobian_determinant;
          VectorizedArray<Number> rhs[n_stages];
          for (unsigned int stage = 0; stage < n_stages; ++stage)
            rhs[stage] = src_ptr[q + stage * dofs_per_component];
          for (unsigned int stage = 0; stage < n_stages; ++stage)
            {
              VectorizedArray<Number> sum = myA[stage][0] * rhs[0];
              for (unsigned int s = 1; s < n_stages; ++s)
                sum += myA[stage][s] * rhs[s];
              dst_ptr[q + stage * dofs_per_component] = sum * factor;
            }
        }
    }

  private:
    std::vector<Number>       inverse_quadrature_weight;
    VectorizedArray<Number>   inverse_jacobian_determinant;
    const FullMatrix<double> &A;
  };



  template <int dim, int fe_degree, int n_stages, typename Number = double>
  class CellwisePreconditionerFDM
  {
  public:
    static constexpr unsigned int n = fe_degree + 1;
    using vcomplex                  = std::complex<VectorizedArray<Number>>;

    CellwisePreconditionerFDM(const IRK &irk)
      : irk(irk)
    {}

    void
    reinit(const std::array<FullMatrix<std::complex<double>>, 2> &eigenvectors,
           const std::array<FullMatrix<std::complex<double>>, 2>
             &inverse_eigenvectors,
           const std::array<std::vector<std::complex<double>>, 2> &eigenvalues,
           const VectorizedArray<Number> inv_jacobian_determinant,
           const Tensor<1, dim, VectorizedArray<Number>> &average_velocity,
           const double                                   inv_dt)
    {
      Tensor<1, dim, VectorizedArray<Number>> blend_factor_eig;
      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
          if (average_velocity[d][v] < 0.0)
            blend_factor_eig[d][v] = 1.0;
          else
            blend_factor_eig[d][v] = 0.0;

      std::array<VectorizedArray<Number>, 2 * n> tmp_eig_x;
      for (unsigned int i0 = 0; i0 < n; ++i0)
        {
          tmp_eig_x[2 * i0] =
            average_velocity[0] *
            ((1.0 - blend_factor_eig[0]) * eigenvalues[0][i0].real() +
             blend_factor_eig[0] * eigenvalues[1][i0].real());
          tmp_eig_x[2 * i0 + 1] =
            average_velocity[0] *
            ((1.0 - blend_factor_eig[0]) * eigenvalues[0][i0].imag() +
             blend_factor_eig[0] * eigenvalues[1][i0].imag());
        }
      std::array<std::complex<Number>, n_stages> mass_contribution;
      for (unsigned int s = 0; s < n_stages; ++s)
        mass_contribution[s] = inv_dt * irk.eigenvalues[s];
      for (unsigned int i2 = 0, c = 0; i2 < (dim > 2 ? n : 1); ++i2)
        for (unsigned int i1 = 0; i1 < (dim > 1 ? n : 1); ++i1)
          {
            vcomplex                    diagonal_element_yz = {};
            std::array<unsigned int, 2> indices{{i1, i2}};
            for (unsigned int d = 1; d < dim; ++d)
              {
                std::complex<VectorizedArray<Number>> eig1(
                  eigenvalues[0][indices[d - 1]]),
                  eig2(eigenvalues[1][indices[d - 1]]);
                diagonal_element_yz +=
                  average_velocity[d] * ((1.0 - blend_factor_eig[d]) * eig1 +
                                         blend_factor_eig[d] * eig2);
              }
            for (unsigned int i0 = 0; i0 < n; ++i0, ++c)
              {
                const vcomplex diagonal_element =
                  diagonal_element_yz +
                  vcomplex(tmp_eig_x[2 * i0], tmp_eig_x[2 * i0 + 1]);
                for (unsigned int s = 0; s < n_stages; ++s)
                  inverse_eigenvalues_for_cell[s][c] =
                    inv_jacobian_determinant /
                    (diagonal_element + vcomplex(mass_contribution[s]));
              }
          }
      for (unsigned int d = 0; d < dim; ++d)
        {
          for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
              {
                {
                  const vcomplex eig1(eigenvectors[0](j, i)),
                    eig2(eigenvectors[1](j, i));
                  this->eigenvectors[d][j * n + i] =
                    (1.0 - blend_factor_eig[d]) * eig1 +
                    blend_factor_eig[d] * eig2;
                }
                {
                  const vcomplex eig1(inverse_eigenvectors[0](j, i)),
                    eig2(inverse_eigenvectors[1](j, i));
                  this->inverse_eigenvectors[d][j * n + i] =
                    (1.0 - blend_factor_eig[d]) * eig1 +
                    blend_factor_eig[d] * eig2;
                }
              }
        }
    }

    void
    vmult(Vector<Number> &dst, const Vector<Number> &src) const
    {
      constexpr unsigned int n_lanes = VectorizedArray<Number>::size();
      constexpr unsigned int n_dofs  = Utilities::pow(n, dim);
      // copy from real to complex vector and apply transformation
      AssertDimension(n_lanes * data_array[0].size() * n_stages, dst.size());
      AssertDimension(n_lanes * data_array[0].size() * n_stages, src.size());
      for (unsigned int i = 0; i < data_array[0].size(); ++i)
        {
          std::array<VectorizedArray<Number>, n_stages> vec_values;
          for (unsigned int s = 0; s < n_stages; ++s)
            vec_values[s].load(src.begin() + i * n_lanes +
                               s * n_lanes * n_dofs);
          for (unsigned int s = 0; s < n_stages; ++s)
            {
              std::complex<VectorizedArray<Number>> sum = {};
              for (unsigned int t = 0; t < n_stages; ++t)
                {
                  std::complex<VectorizedArray<Number>> mat_entry;
                  mat_entry.real(
                    make_vectorized_array<Number>(irk.inv_T(s, t).real()));
                  mat_entry.imag(
                    make_vectorized_array<Number>(irk.inv_T(s, t).imag()));
                  sum += mat_entry * vec_values[t];
                }
              data_array[s][i] = sum;
            }
        }

      using Eval = internal::
        EvaluatorTensorProduct<internal::evaluate_general, dim, n, n, vcomplex>;
      for (unsigned int s = 0; s < n_stages; ++s)
        {
          // apply V^{-1} M^{-1}
          Eval::template apply<0, false, false>(inverse_eigenvectors[0].data(),
                                                data_array[s].data(),
                                                data_array[s].data());
          if (dim > 1)
            Eval::template apply<1, false, false>(
              inverse_eigenvectors[1].data(),
              data_array[s].data(),
              data_array[s].data());
          if (dim > 2)
            Eval::template apply<2, false, false>(
              inverse_eigenvectors[2].data(),
              data_array[s].data(),
              data_array[s].data());

          // apply inv(I x Lambda + Lambda x I)
          for (unsigned int c = 0; c < data_array[0].size(); ++c)
            data_array[s][c] *= inverse_eigenvalues_for_cell[s][c];

          // apply V
          Eval::template apply<0, false, false>(eigenvectors[0].data(),
                                                data_array[s].data(),
                                                data_array[s].data());
          if (dim > 1)
            Eval::template apply<1, false, false>(eigenvectors[1].data(),
                                                  data_array[s].data(),
                                                  data_array[s].data());
          if (dim > 2)
            Eval::template apply<2, false, false>(eigenvectors[2].data(),
                                                  data_array[s].data(),
                                                  data_array[s].data());
        }

      // copy back to real vector, also applying the transformation of the IRK
      // time stepping
      for (unsigned int i = 0; i < data_array[0].size(); ++i)
        {
          for (unsigned int s = 0; s < n_stages; ++s)
            {
              std::complex<VectorizedArray<Number>> sum = {};
              for (unsigned int t = 0; t < n_stages; ++t)
                {
                  std::complex<VectorizedArray<Number>> mat_entry;
                  mat_entry.real(
                    make_vectorized_array<Number>(irk.T(s, t).real()));
                  mat_entry.imag(
                    make_vectorized_array<Number>(irk.T(s, t).imag()));
                  sum += mat_entry * data_array[t][i];
                }
              sum.real().store(dst.begin() + i * n_lanes +
                               s * n_lanes * n_dofs);
            }
        }
    }

  private:
    dealii::ndarray<vcomplex, dim, n * n> eigenvectors;
    dealii::ndarray<vcomplex, dim, n * n> inverse_eigenvectors;
    dealii::ndarray<vcomplex, n_stages, Utilities::pow(n, dim)>
      inverse_eigenvalues_for_cell;
    mutable dealii::ndarray<vcomplex, n_stages, Utilities::pow(n, dim)>
               data_array;
    const IRK &irk;
  };



  // Implementation of the advection operation
  template <int dim, int fe_degree>
  class AdvectionOperation
  {
  public:
    using Number = double;

    AdvectionOperation()
      : computing_times(4)
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
            Utilities::MPI::min_max_avg(computing_times[1], MPI_COMM_WORLD);
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "Time rhs compute (min / avg / max): " << data.min
                      << " (proc_" << data.min_index << ") / " << data.avg
                      << " / " << data.max << " (proc_" << data.max_index << ")"
                      << std::endl;
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
    compute_rhs(LinearAlgebra::distributed::BlockVector<Number>  &dst,
                const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      Timer time;
      data.loop(&AdvectionOperation<dim, fe_degree>::local_rhs_domain,
                &AdvectionOperation<dim, fe_degree>::local_rhs_inner_face,
                &AdvectionOperation<dim, fe_degree>::local_rhs_boundary_face,
                this,
                dst,
                src,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
      dst *= -1. / time_step;
      computing_times[1] += time.wall_time();
    }

    void
    update_solution(LinearAlgebra::distributed::Vector<Number> &solution,
                    const LinearAlgebra::distributed::BlockVector<Number>
                      &stage_solution) const
    {
      Timer time;
      DEAL_II_OPENMP_SIMD_PRAGMA
      for (unsigned int i = 0; i < stage_solution.block(0).locally_owned_size();
           ++i)
        {
          std::array<Number, n_stages> values;
          for (unsigned int s = 0; s < n_stages; ++s)
            values[s] = stage_solution.block(s).local_element(i);
          Number my_sol = 0.;
          for (unsigned int c = 0; c < n_stages; ++c)
            {
              Number sum = 0.;
              for (unsigned int b = 0; b < n_stages; ++b)
                sum += values[b] * irk.inv_A(c, b);
              my_sol += irk.b(c) * sum;
            }
          solution.local_element(i) += time_step * my_sol;
        }
      computing_times[1] += time.wall_time();
    }

    void
    vmult(LinearAlgebra::distributed::BlockVector<Number>       &dst,
          const LinearAlgebra::distributed::BlockVector<Number> &src) const
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
    precondition_block_jacobi(
      LinearAlgebra::distributed::BlockVector<Number>       &dst,
      const LinearAlgebra::distributed::BlockVector<Number> &src) const;

    void
    project_initial(LinearAlgebra::distributed::Vector<Number> &dst) const;

    Tensor<1, 3>
    compute_mass_and_energy(
      const LinearAlgebra::distributed::Vector<Number> &vec) const;

  private:
    MatrixFree<dim, Number> data;
    double                  time;
    double                  time_step;
    IRK                     irk;

    mutable std::vector<double> computing_times;

    Table<2, Tensor<1, dim, VectorizedArray<Number>>> speeds_cells;
    Table<2, Tensor<1, dim, VectorizedArray<Number>>> speeds_faces;
    std::vector<Table<2, VectorizedArray<Number>>>    normal_speeds_faces;

    std::array<FullMatrix<std::complex<double>>, 2> eigenvectors,
      inverse_eigenvectors;
    std::array<std::vector<std::complex<double>>, 2>       eigenvalues;
    AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> scaled_cell_velocity;

    void
    local_apply_domain(
      const MatrixFree<dim, Number>                         &data,
      LinearAlgebra::distributed::BlockVector<Number>       &dst,
      const LinearAlgebra::distributed::BlockVector<Number> &src,
      const std::pair<unsigned int, unsigned int>           &cell_range) const;

    void
    local_apply_inner_face(
      const MatrixFree<dim, Number>                         &data,
      LinearAlgebra::distributed::BlockVector<Number>       &dst,
      const LinearAlgebra::distributed::BlockVector<Number> &src,
      const std::pair<unsigned int, unsigned int>           &cell_range) const;
    void
    local_apply_boundary_face(
      const MatrixFree<dim, Number>                         &data,
      LinearAlgebra::distributed::BlockVector<Number>       &dst,
      const LinearAlgebra::distributed::BlockVector<Number> &src,
      const std::pair<unsigned int, unsigned int>           &cell_range) const;

    void
    local_rhs_domain(
      const MatrixFree<dim, Number>                    &data,
      LinearAlgebra::distributed::BlockVector<Number>  &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int>      &cell_range) const;

    void
    local_rhs_inner_face(
      const MatrixFree<dim, Number>                    &data,
      LinearAlgebra::distributed::BlockVector<Number>  &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int>      &cell_range) const;
    void
    local_rhs_boundary_face(
      const MatrixFree<dim, Number>                    &data,
      LinearAlgebra::distributed::BlockVector<Number>  &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int>      &cell_range) const;
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
                std::vector<const DoFHandler<dim> *>{&dof_handler},
                std::vector<const AffineConstraints<double> *>{&dummy},
                std::vector<Quadrature<1>>{{quadrature, quadrature_mass}},
                additional_data);

    // precompute spatial part of variable advection speed, where we utilize
    // that the scaling in time is 1 for time t=0
    TransportSpeed<dim> transport_speed(0);
    {
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval(data);
      speeds_cells.reinit(data.n_cell_batches(), eval.n_q_points);
      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
        {
          eval.reinit(cell);
          for (unsigned int q = 0; q < eval.n_q_points; ++q)
            speeds_cells(cell, q) =
              transport_speed.value(eval.quadrature_point(q));
        }
    }
    {
      FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval(data,
                                                                      true);
      speeds_faces.reinit(data.n_inner_face_batches() +
                            data.n_boundary_face_batches(),
                          eval.n_q_points);
      for (unsigned int face = 0;
           face < data.n_inner_face_batches() + data.n_boundary_face_batches();
           ++face)
        {
          eval.reinit(face);
          for (unsigned int q = 0; q < eval.n_q_points; ++q)
            speeds_faces(face, q) =
              transport_speed.value(eval.quadrature_point(q));
        }

      FEFaceValues<dim> fe_face_values(mapping,
                                       dof_handler.get_fe(),
                                       QGauss<dim - 1>(fe_degree + 1),
                                       update_normal_vectors |
                                         update_quadrature_points);
      normal_speeds_faces.resize(data.n_cell_batches());
      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
        {
          normal_speeds_faces[cell].reinit(2 * dim, eval.n_q_points);
          for (unsigned int v = 0;
               v < data.n_active_entries_per_cell_batch(cell);
               ++v)
            for (unsigned int f = 0; f < 2 * dim; ++f)
              {
                fe_face_values.reinit(data.get_cell_iterator(cell, v), f);
                for (unsigned int q = 0; q < eval.n_q_points; ++q)
                  normal_speeds_faces[cell][f][q][v] =
                    transport_speed.value(fe_face_values.quadrature_point(q)) *
                    fe_face_values.normal_vector(q);
              }
        }
    }

    QGauss<1>               gauss_quad(dof_handler.get_fe().degree + 1);
    FE_DGQArbitraryNodes<1> fe_1d(gauss_quad);
    constexpr unsigned int  n = fe_degree + 1;
    for (unsigned int c = 0; c < 2; ++c)
      {
        LAPACKFullMatrix<double> deriv_matrix(n, n);
        for (unsigned int q = 0; q < n; ++q)
          {
            for (unsigned int i = 0; i < n; ++i)
              for (unsigned int j = 0; j < n; ++j)
                deriv_matrix(i, j) -=
                  fe_1d.shape_grad(i, gauss_quad.point(q))[0] *
                  fe_1d.shape_value(j, gauss_quad.point(q)) *
                  gauss_quad.weight(q);
          }
        const double sign_advection = (c == 0) ? 1.0 : -1.0;
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            deriv_matrix(i, j) += -fe_1d.shape_value(i, Point<1>()) *
                                    fe_1d.shape_value(j, Point<1>()) *
                                    (0.5 - flux_alpha * 0.5 * sign_advection) +
                                  fe_1d.shape_value(i, Point<1>(1.0)) *
                                    fe_1d.shape_value(j, Point<1>(1.0)) *
                                    (0.5 + flux_alpha * 0.5 * sign_advection);

        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            deriv_matrix(i, j) *= (1. / gauss_quad.weight(i));
        deriv_matrix.compute_eigenvalues(true, false);

        eigenvalues[c].resize(n);
        for (unsigned int i = 0; i < n; ++i)
          eigenvalues[c][i] = deriv_matrix.eigenvalue(i);

        eigenvectors[c]         = deriv_matrix.get_right_eigenvectors();
        inverse_eigenvectors[c] = eigenvectors[c];
        inverse_eigenvectors[c].gauss_jordan();
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            inverse_eigenvectors[c](i, j) *= (1. / gauss_quad.weight(j));
      }
    scaled_cell_velocity.resize(data.n_cell_batches());
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval(data);
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        eval.reinit(cell);
        Tensor<1, dim, VectorizedArray<Number>> average_velocity;
        VectorizedArray<Number>                 cell_volume = {};
        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            average_velocity +=
              eval.inverse_jacobian(q) * speeds_cells(cell, q) * eval.JxW(q);
            cell_volume += eval.JxW(q);
          }
        scaled_cell_velocity[cell] = average_velocity / cell_volume;
      }

    /*
    this->time_step = 0.1;
    IRK                          irk;
    std::array<Number, n_stages> factor_time;
    for (unsigned int s = 0; s < n_stages; ++s)
      factor_time[s] =
        std::cos(numbers::PI * (time + irk.c(s) * time_step) / FINAL_TIME);

    CellwiseOperator<dim, fe_degree, n_stages> inversion(
      data.get_mapping_info().cell_data[0].jacobians[0][0],
      data.get_shape_info().data[0],
      &speeds_cells(0, 0),
      normal_speeds_faces[0],
      data.get_mapping_info().cell_data[0].descriptor[0].quadrature,
      data.get_mapping_info().face_data[0].descriptor[0].quadrature,
      1. / time_step,
      irk.inv_A,
      factor_time);

    const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
    Vector<double>     local(dofs_per_cell * VectorizedArray<Number>::size() *
                         n_stages);
    Vector<double>     rhs(local);
    FullMatrix<double> mat(dofs_per_cell * n_stages, dofs_per_cell * n_stages);
    for (unsigned int i = 0; i < mat.m(); ++i)
      {
        rhs                                      = 0;
        rhs(i * VectorizedArray<Number>::size()) = 1.;
        inversion.vmult(local, rhs);
        for (unsigned int j = 0; j < mat.m(); ++j)
          mat(j, i) = local(j * VectorizedArray<Number>::size());
      }
    mat.print_formatted(std::cout);

    std::cout << std::endl;

    LinearAlgebra::distributed::BlockVector<double> sol(n_stages),
      second(n_stages);
    for (unsigned int s = 0; s < n_stages; ++s)
      data.initialize_dof_vector(sol.block(s));
    sol.collect_sizes();
    second.reinit(sol);
    for (unsigned int i = 0; i < mat.m(); ++i)
      {
        second = 0;
        if (i < dofs_per_cell)
          second.block(0)(i) = 1.;
        else
          second.block(1)(i - dofs_per_cell) = 1.;

        this->vmult(sol, second);
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          mat(j, i) = sol.block(0)(j);
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          mat(j + dofs_per_cell, i) = sol.block(1)(j);
      }
    mat.print_formatted(std::cout);
    */
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::local_apply_domain(
    const MatrixFree<dim, Number>                         &data,
    LinearAlgebra::distributed::BlockVector<Number>       &dst,
    const LinearAlgebra::distributed::BlockVector<Number> &src,
    const std::pair<unsigned int, unsigned int>           &cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, n_stages, Number> eval(data);
    const double inv_dt = 1. / time_step;

    std::array<Number, n_stages> factor_time;
    for (unsigned int s = 0; s < n_stages; ++s)
      factor_time[s] =
        std::cos(numbers::PI * (time + irk.c(s) * time_step) / FINAL_TIME);

    dealii::ndarray<Number, n_stages, n_stages> inv_A_dt;
    for (unsigned int i = 0; i < n_stages; ++i)
      for (unsigned int j = 0; j < n_stages; ++j)
        inv_A_dt[i][j] = irk.inv_A(i, j) * inv_dt;

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);

        // compute u^h(x) from src
        eval.gather_evaluate(src,
                             EvaluationFlags::values |
                               EvaluationFlags::gradients);

        // loop over quadrature points and compute the local volume flux
        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            const auto speed = speeds_cells(cell, q);
            const auto u     = eval.get_value(q);
            const auto gradu = eval.get_gradient(q);
            Tensor<1, n_stages, Tensor<1, dim, VectorizedArray<Number>>>
              volume_flux;
            for (unsigned int c = 0; c < n_stages; ++c)
              volume_flux[c] =
                ((-1.0 + factor_skew) * speed) * (factor_time[c] * u[c]);
            if constexpr (n_stages == 1)
              {
                eval.submit_gradient(volume_flux[0], q);
                eval.submit_value(((factor_skew * speed) * gradu) *
                                      factor_time[0] +
                                    inv_dt * u[0],
                                  q);
              }
            else
              {
                eval.submit_gradient(volume_flux, q);
                Tensor<1, n_stages, VectorizedArray<Number>> volume_val;
                for (unsigned int c = 0; c < n_stages; ++c)
                  {
                    volume_val[c] =
                      ((factor_skew * speed) * gradu[c]) * factor_time[c];
                    for (unsigned int b = 0; b < n_stages; ++b)
                      volume_val[c] += inv_A_dt[c][b] * u[b];
                  }
                eval.submit_value(volume_val, q);
              }
          }

        // multiply by nabla v^h(x) and sum
        eval.integrate_scatter(EvaluationFlags::values |
                                 EvaluationFlags::gradients,
                               dst);
      }
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::local_apply_inner_face(
    const MatrixFree<dim, Number>                         &data,
    LinearAlgebra::distributed::BlockVector<Number>       &dst,
    const LinearAlgebra::distributed::BlockVector<Number> &src,
    const std::pair<unsigned int, unsigned int>           &face_range) const
  {
    // On interior faces, we have two evaluators, one for the solution
    // 'u_minus' and one for the solution 'u_plus'. Note that the decision
    // about what is minus and plus is arbitrary at this point, so we must
    // assume that this can be arbitrarily oriented and we must only operate
    // with the generic quantities such as the normal vector.
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, n_stages, Number>
      eval_minus(data, true);
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, n_stages, Number> eval_plus(
      data, false);

    std::array<Number, n_stages> factor_time;
    for (unsigned int s = 0; s < n_stages; ++s)
      factor_time[s] =
        std::cos(numbers::PI * (time + irk.c(s) * time_step) / FINAL_TIME);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_plus.reinit(face);
        eval_minus.gather_evaluate(src, EvaluationFlags::values);
        eval_plus.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const auto speed               = speeds_faces(face, q);
            const auto u_minus             = eval_minus.get_value(q);
            const auto u_plus              = eval_plus.get_value(q);
            const auto normal_vector_minus = eval_minus.normal_vector(q);

            Tensor<1, n_stages, VectorizedArray<Number>> flux_minus;
            Tensor<1, n_stages, VectorizedArray<Number>> flux_plus;
            for (unsigned int c = 0; c < n_stages; ++c)
              {
                const auto normal_times_speed =
                  (speed * normal_vector_minus) * factor_time[c];
                const auto flux_times_normal_of_u_minus =
                  0.5 * ((u_minus[c] + u_plus[c]) * normal_times_speed +
                         flux_alpha * std::abs(normal_times_speed) *
                           (u_minus[c] - u_plus[c]));
                flux_minus[c] = flux_times_normal_of_u_minus -
                                factor_skew * normal_times_speed * u_minus[c];
                flux_plus[c] = -flux_times_normal_of_u_minus +
                               factor_skew * normal_times_speed * u_plus[c];
              }

            eval_minus.submit_value(flux_minus, q);
            eval_plus.submit_value(flux_plus, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values, dst);
        eval_plus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::local_apply_boundary_face(
    const MatrixFree<dim, Number>                         &data,
    LinearAlgebra::distributed::BlockVector<Number>       &dst,
    const LinearAlgebra::distributed::BlockVector<Number> &src,
    const std::pair<unsigned int, unsigned int>           &face_range) const
  {
    AssertThrow(false, ExcNotImplemented());
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, n_stages, Number>
                                 eval_minus(data, true);
    ExactSolution<dim>           solution(time + irk.c(0) * time_step);
    std::array<Number, n_stages> factor_time;
    for (unsigned int s = 0; s < n_stages; ++s)
      factor_time[s] =
        std::cos(numbers::PI * (time + irk.c(s) * time_step) / FINAL_TIME);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_minus.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const auto speed = speeds_faces(face, q);
            // Dirichlet boundary
            const auto u_minus       = eval_minus.get_value(q);
            const auto normal_vector = eval_minus.normal_vector(q);

            // Compute the outer solution value
            Tensor<1, n_stages, VectorizedArray<Number>> flux;
            for (unsigned int c = 0; c < n_stages; ++c)
              {
                const auto u_plus =
                  solution.value(eval_minus.quadrature_point(q));

                // compute the flux
                const auto normal_times_speed =
                  (normal_vector * speed) * factor_time[c];
                const auto flux_times_normal =
                  0.5 * ((u_minus[c] + u_plus) * normal_times_speed +
                         flux_alpha * std::abs(normal_times_speed) *
                           (u_minus[c] - u_plus));

                flux[c] = flux_times_normal -
                          factor_skew * normal_times_speed * u_minus[c];
              }
            eval_minus.submit_value(flux, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::local_rhs_domain(
    const MatrixFree<dim, Number>                    &data,
    LinearAlgebra::distributed::BlockVector<Number>  &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int>      &cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, n_stages, Number> eval(data);
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_src(data);

    std::array<Number, n_stages> factor_time;
    for (unsigned int s = 0; s < n_stages; ++s)
      factor_time[s] =
        std::cos(numbers::PI * (time + irk.c(s) * time_step) / FINAL_TIME);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval_src.reinit(cell);

        // compute u^h(x) from src
        eval_src.gather_evaluate(src,
                                 EvaluationFlags::values |
                                   EvaluationFlags::gradients);

        // loop over quadrature points and compute the local volume flux
        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            const auto speed = speeds_cells(cell, q);
            const auto u     = eval_src.get_value(q);
            const auto gradu = eval_src.get_gradient(q);
            Tensor<1, n_stages, Tensor<1, dim, VectorizedArray<Number>>>
              volume_flux;
            for (unsigned int c = 0; c < n_stages; ++c)
              volume_flux[c] =
                ((-1.0 + factor_skew) * speed * u) * factor_time[c];
            if constexpr (n_stages == 1)
              eval.submit_gradient(volume_flux[0], q);
            else
              eval.submit_gradient(volume_flux, q);
            Tensor<1, n_stages, VectorizedArray<Number>> volume_val;
            for (unsigned int c = 0; c < n_stages; ++c)
              {
                volume_val[c] =
                  (factor_skew * (speed * gradu)) * factor_time[c];
              }
            eval.submit_value(volume_val, q);
          }

        // multiply by nabla v^h(x) and sum
        eval.integrate_scatter(EvaluationFlags::values |
                                 EvaluationFlags::gradients,
                               dst);
      }
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::local_rhs_inner_face(
    const MatrixFree<dim, Number>                    &data,
    LinearAlgebra::distributed::BlockVector<Number>  &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int>      &face_range) const
  {
    // On interior faces, we have two evaluators, one for the solution
    // 'u_minus' and one for the solution 'u_plus'. Note that the decision
    // about what is minus and plus is arbitrary at this point, so we must
    // assume that this can be arbitrarily oriented and we must only operate
    // with the generic quantities such as the normal vector.
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, n_stages, Number>
      eval_minus(data, true);
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, n_stages, Number> eval_plus(
      data, false);
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_src_minus(
      data, true);
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_src_plus(
      data, false);

    std::array<Number, n_stages> factor_time;
    for (unsigned int s = 0; s < n_stages; ++s)
      factor_time[s] =
        std::cos(numbers::PI * (time + irk.c(s) * time_step) / FINAL_TIME);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_plus.reinit(face);
        eval_src_minus.reinit(face);
        eval_src_plus.reinit(face);
        eval_src_minus.gather_evaluate(src, EvaluationFlags::values);
        eval_src_plus.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const auto speed               = speeds_faces(face, q);
            const auto u_minus             = eval_src_minus.get_value(q);
            const auto u_plus              = eval_src_plus.get_value(q);
            const auto normal_vector_minus = eval_minus.normal_vector(q);

            Tensor<1, n_stages, VectorizedArray<Number>> flux_minus;
            Tensor<1, n_stages, VectorizedArray<Number>> flux_plus;
            for (unsigned int c = 0; c < n_stages; ++c)
              {
                const auto normal_times_speed =
                  (speed * normal_vector_minus) * factor_time[c];
                const auto flux_times_normal_of_u_minus =
                  0.5 * ((u_minus + u_plus) * normal_times_speed +
                         flux_alpha * std::abs(normal_times_speed) *
                           (u_minus - u_plus));
                flux_minus[c] = flux_times_normal_of_u_minus -
                                factor_skew * normal_times_speed * u_minus;
                flux_plus[c] = -flux_times_normal_of_u_minus +
                               factor_skew * normal_times_speed * u_plus;
              }

            eval_minus.submit_value(flux_minus, q);
            eval_plus.submit_value(flux_plus, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values, dst);
        eval_plus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::local_rhs_boundary_face(
    const MatrixFree<dim, Number>                    &data,
    LinearAlgebra::distributed::BlockVector<Number>  &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, n_stages, Number>
      eval_minus(data, true);
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_src_minus(
      data, true);

    ExactSolution<dim>           solution(time + irk.c(0) * time_step);
    std::array<Number, n_stages> factor_time;
    for (unsigned int s = 0; s < n_stages; ++s)
      factor_time[s] =
        std::cos(numbers::PI * (time + irk.c(s) * time_step) / FINAL_TIME);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_src_minus.reinit(face);
        eval_src_minus.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const auto speed = speeds_faces(face, q);
            // Dirichlet boundary
            const auto u_minus       = eval_src_minus.get_value(q);
            const auto normal_vector = eval_minus.normal_vector(q);

            // Compute the outer solution value
            Tensor<1, n_stages, VectorizedArray<Number>> flux;
            for (unsigned int c = 0; c < n_stages; ++c)
              {
                const auto u_plus =
                  solution.value(eval_minus.quadrature_point(q));

                // compute the flux
                const auto normal_times_speed =
                  (normal_vector * speed) * factor_time[c];
                const auto flux_times_normal =
                  0.5 * ((u_minus + u_plus) * normal_times_speed +
                         flux_alpha * std::abs(normal_times_speed) *
                           (u_minus - u_plus));

                flux[c] = flux_times_normal -
                          factor_skew * normal_times_speed * u_minus;
              }
            eval_minus.submit_value(flux, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values, dst);
      }
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



  template <typename VectorType>
  class MyVectorMemory : public VectorMemory<VectorType>
  {
  public:
    MyVectorMemory()
      : first_unused(vectors.end())
    {}

    virtual VectorType *
    alloc() override
    {
      if (first_unused == vectors.end())
        {
          vectors.push_back(VectorType());
          return &vectors.back();
        }
      else
        {
          VectorType *return_value = &(*first_unused);
          ++first_unused;
          return return_value;
        }
    }

    virtual void
    free(const VectorType *const vector) override
    {
      typename std::list<VectorType>::iterator it = vectors.begin();
      while (&*it != vector)
        ++it;

      Assert(it != first_unused && vector == &*it, ExcInternalError());
      vectors.splice(first_unused, vectors, it);
      --first_unused;
    }

  private:
    std::list<VectorType>                    vectors;
    typename std::list<VectorType>::iterator first_unused;
  };


  template <int dim, int fe_degree>
  void
  AdvectionOperation<dim, fe_degree>::precondition_block_jacobi(
    LinearAlgebra::distributed::BlockVector<Number>       &dst,
    const LinearAlgebra::distributed::BlockVector<Number> &src) const
  {
    Timer                        timer;
    std::array<Number, n_stages> factor_time;
    for (unsigned int s = 0; s < n_stages; ++s)
      factor_time[s] =
        std::cos(numbers::PI * (time + irk.c(s) * time_step) / FINAL_TIME);

    FEEvaluation<dim, fe_degree, fe_degree + 1, n_stages, Number> eval(data);
    // GrowingVectorMemory<Vector<double>>                           memory;
    MyVectorMemory<Vector<double>> memory;
    Vector<double>                 local_src(eval.dofs_per_cell *
                             VectorizedArray<Number>::size());
    Vector<double>                 local_dst(local_src);

    /*
    eval.reinit(0);
    CellwiseOperator<dim, fe_degree, n_stages> local_operator(
      eval.inverse_jacobian(0),
      data.get_shape_info().data[0],
      &speeds_cells(0, 0),
      normal_speeds_faces[0],
      data.get_mapping_info().cell_data[0].descriptor[0].quadrature,
      data.get_mapping_info().face_data[0].descriptor[0].quadrature,
      1. / time_step,
      irk.inv_A,
      {1.0}); // factor_time);
    for (unsigned int d = 0; d < eval.dofs_per_cell; ++d)
      {
        local_src                                      = 0.;
        local_src(d * VectorizedArray<Number>::size()) = 1.;
        local_operator.vmult(local_dst, local_src);
        for (unsigned int e = 0; e < eval.dofs_per_cell; ++e)
          std::cout << local_dst(e * VectorizedArray<Number>::size()) << " ";
        std::cout << std::endl;
      }
    std::cout << std::endl;
    CellwisePreconditionerFDM<dim, fe_degree, n_stages, Number> precond(
      eigenvectors,
      inverse_eigenvectors,
      eigenvalues,
      determinant(eval.inverse_jacobian(0)),
      scaled_cell_velocity[0],
      1. / time_step,
      irk,
      {1.0});
    for (unsigned int d = 0; d < eval.dofs_per_cell; ++d)
      {
        local_src                                      = 0.;
        local_src(d * VectorizedArray<Number>::size()) = 1.;
        precond.vmult(local_dst, local_src);
        for (unsigned int e = 0; e < eval.dofs_per_cell; ++e)
          std::cout << local_dst(e * VectorizedArray<Number>::size()) << " ";
        std::cout << std::endl;
      }
    std::cout << std::endl;
    std::abort();
    */

    // CellwisePreconditioner<n_stages, Number> precondition(
    //  data.get_mapping_info().cell_data[0].descriptor[0].quadrature, irk.A);
    const unsigned int     n_max_iterations = 4;
    IterationNumberControl control(n_max_iterations, 1e-18, false, false);
    typename SolverGMRES<Vector<Number>>::AdditionalData gmres_data;
    gmres_data.right_preconditioning = true;
    gmres_data.orthogonalization_strategy =
      LinearAlgebra::OrthogonalizationStrategy::classical_gram_schmidt;
    gmres_data.max_basis_size = n_max_iterations;
    gmres_data.batched_mode   = true;
    // gmres_data.exact_residual = false;
    SolverGMRES<Vector<Number>> gmres(control, memory, gmres_data);
    CellwisePreconditionerFDM<dim, fe_degree, n_stages> precondition(irk);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        eval.reinit(cell);
        eval.read_dof_values(src);
        CellwiseOperator<dim, fe_degree, n_stages> local_operator(
          eval.inverse_jacobian(0),
          data.get_shape_info().data[0],
          &speeds_cells(cell, 0),
          normal_speeds_faces[cell],
          data.get_mapping_info().cell_data[0].descriptor[0].quadrature,
          data.get_mapping_info().face_data[0].descriptor[0].quadrature,
          1. / time_step,
          irk.inv_A,
          factor_time);
        double avg_time_factor = factor_time[0];
        for (unsigned int s = 1; s < n_stages; ++s)
          avg_time_factor += factor_time[s];
        avg_time_factor *= (1. / n_stages);
        precondition.reinit(eigenvectors,
                            inverse_eigenvectors,
                            eigenvalues,
                            determinant(eval.inverse_jacobian(0)),
                            scaled_cell_velocity[cell] * avg_time_factor,
                            1. / time_step);
        local_operator.transform_to_collocation(eval.begin_dof_values(),
                                                local_src);
        local_dst = 0;
        gmres.solve(local_operator, local_dst, local_src, precondition);
        // precondition.vmult(local_dst, local_src);
        local_operator.transform_from_collocation(local_dst,
                                                  eval.begin_dof_values());
        /*
        CellwiseRungeKutta<dim, fe_degree, n_stages> local_rk(
          eval.inverse_jacobian(0),
          data.get_shape_info().data[0],
          &speeds_cells(cell, 0),
          normal_speeds_faces[cell],
          data.get_mapping_info().cell_data[0].descriptor[0].quadrature,
          data.get_mapping_info().face_data[0].descriptor[0].quadrature,
          time_step,
          irk.A,
          factor_time);
        local_rk.transform_to_collocation(eval.begin_dof_values(),
                                          eval.begin_dof_values());
        local_rk.propagate(eval.begin_dof_values());
        //for (unsigned int i=0; i<eval.dofs_per_cell; ++i)
        //  std::cout << eval.begin_dof_values()[i] << " ";
        //std::cout << std::endl;
        local_rk.transform_from_collocation(eval.begin_dof_values(),
                                                  eval.begin_dof_values());
        */
        eval.set_dof_values(dst);
      }
    computing_times[3] += timer.wall_time();
  }



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
    time_step = courant_number * glob_min_vertex_distance / 2.;

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


  template <int dim, int fe_degree>
  class Precondition
  {
  public:
    Precondition(const MatrixFree<dim, double> &data)
    {
      data.initialize_dof_vector(inverse_diagonal);
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> eval(data);
      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
        {
          eval.reinit(cell);
          for (unsigned int q = 0; q < eval.n_q_points; ++q)
            eval.submit_value(1.0, q);
          eval.integrate(EvaluationFlags::values);
          for (unsigned int i = 0; i < eval.dofs_per_cell; ++i)
            eval.begin_dof_values()[i] = 1.0 / eval.begin_dof_values()[i];
          eval.set_dof_values(inverse_diagonal);
        }
    }

    void
    set_time_step(const double time_step)
    {
      this->time_step = time_step;
    }

    void
    vmult(LinearAlgebra::distributed::BlockVector<double>       &dst,
          const LinearAlgebra::distributed::BlockVector<double> &src) const
    {
      IRK irk;
      DEAL_II_OPENMP_SIMD_PRAGMA
      for (unsigned int i = 0; i < src.block(0).locally_owned_size(); ++i)
        {
          std::array<double, n_stages> values;
          for (unsigned int s = 0; s < n_stages; ++s)
            values[s] = src.block(s).local_element(i);
          const double factor = inverse_diagonal.local_element(i) * time_step;
          for (unsigned int c = 0; c < n_stages; ++c)
            {
              double sum = 0.;
              for (unsigned int b = 0; b < n_stages; ++b)
                sum += values[b] * irk.A(c, b);
              dst.block(c).local_element(i) = factor * sum;
            }
        }
    }

  private:
    LinearAlgebra::distributed::Vector<double> inverse_diagonal;
    double                                     time_step;
  };

  template <typename OperatorType>
  class BlockJacobi
  {
  public:
    BlockJacobi(const OperatorType &operator_exemplar)
      : operator_exemplar(operator_exemplar)
    {}

    void
    vmult(LinearAlgebra::distributed::BlockVector<double>       &dst,
          const LinearAlgebra::distributed::BlockVector<double> &src) const
    {
      operator_exemplar.precondition_block_jacobi(dst, src);
    }

  private:
    const OperatorType &operator_exemplar;
  };


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
    /*
    const auto multiple_part =
      create_partitioner_multiple(solution.get_partitioner(), 4);
    LinearAlgebra::distributed::Vector<Number> sol2(multiple_part);
    pcout << "Vector sizes: " << solution.size() << " " << sol2.size() << " "
          << sol2.get_partitioner()->n_ghost_indices() << std::endl;
    */
    advection_operator.project_initial(solution);

    LinearAlgebra::distributed::Vector<Number>      solution_copy = solution;
    LinearAlgebra::distributed::BlockVector<Number> rhs(n_stages);
    for (unsigned int s = 0; s < n_stages; ++s)
      rhs.block(s).reinit(solution);
    rhs.collect_sizes();
    std::vector<LinearAlgebra::distributed::BlockVector<Number>> stage_sol(5,
                                                                           rhs);
    std::vector<LinearAlgebra::distributed::BlockVector<Number>> stage_mv(5,
                                                                          rhs);

    Precondition<dim, fe_degree> precondition_m(
      advection_operator.get_matrix_free());
    precondition_m.set_time_step(time_step);

    BlockJacobi<AdvectionOperation<dim, fe_degree>> precondition(
      advection_operator);

    unsigned int n_output = 0;
    output_results(n_output++,
                   advection_operator.compute_mass_and_energy(solution));
    pcout << std::endl;

    Timer        timer;
    double       prep_time       = 0;
    double       sol_time        = 0;
    double       output_time     = 0;
    unsigned int timestep_number = 1;

    // This is the main time loop, asking the time integrator class to perform
    // the time step and update the content in the solution vector.
    while (time < FINAL_TIME - 1e-12)
      {
        timer.restart();

        advection_operator.set_time(time, time_step);

        advection_operator.compute_rhs(rhs, solution);

        // Compute upper triangular matrix with orthogonal factors of the
        // current matrix applied to old solutions of the linear system,
        // orthogonalized by the modified Gram-Schmidt process
        const unsigned int n_max_steps =
          timestep_number > 5 ? 5 : timestep_number - 1;
        FullMatrix<double> projection_matrix(n_max_steps, n_max_steps);
        unsigned int       step = 0;
        for (; step < n_max_steps; ++step)
          {
            advection_operator.vmult(stage_mv[step], stage_sol[step]);
            projection_matrix(0, step) = stage_mv[step] * stage_mv[0];
            for (unsigned int j = 0; j < step; ++j)
              projection_matrix(j + 1, step) =
                stage_mv[step].add_and_dot(-projection_matrix(j, step) /
                                             projection_matrix(j, j),
                                           stage_mv[j],
                                           stage_mv[j + 1]);

            // Note that the entries in the matrix are the square of the norm,
            // so we request that vectors which are below 1e-8 of being
            // linearly independent are discarded
            if (projection_matrix(step, step) < 1e-16 * projection_matrix(0, 0))
              break;
          }
        // Solve least-squares system
        std::array<double, 5> project_sol = {};
        for (unsigned int s = 0; s < step; ++s)
          project_sol[s] = stage_mv[s] * rhs;
        for (int s = step - 1; s >= 0; --s)
          {
            double sum = project_sol[s];
            for (unsigned int j = s + 1; j < step; ++j)
              sum -= project_sol[j] * projection_matrix(s, j);
            project_sol[s] = sum / projection_matrix(s, s);
          }

        // extrapolate solution from old values
        for (unsigned int bl = 0; bl < stage_sol[0].n_blocks(); ++bl)
          {
            const unsigned int local_size =
              stage_sol[0].block(bl).locally_owned_size();
            std::array<Number *, 5> vec_ptrs;
            for (unsigned int i = 0; i < vec_ptrs.size(); ++i)
              vec_ptrs[i] = stage_sol[i].block(bl).begin();
            DEAL_II_OPENMP_SIMD_PRAGMA
            for (unsigned int i = 0; i < local_size; ++i)
              {
                const double sol_0 = vec_ptrs[0][i];
                const double sol_1 = vec_ptrs[1][i];
                const double sol_2 = vec_ptrs[2][i];
                const double sol_3 = vec_ptrs[3][i];
                const double sol_4 = vec_ptrs[4][i];
                vec_ptrs[0][i] =
                  project_sol[0] * sol_0 + project_sol[1] * sol_1 +
                  project_sol[2] * sol_2 + project_sol[3] * sol_3 +
                  project_sol[4] * sol_4;
                vec_ptrs[1][i] = sol_0;
                vec_ptrs[2][i] = sol_1;
                vec_ptrs[3][i] = sol_2;
                vec_ptrs[4][i] = sol_3;
              }
          }

        prep_time += timer.wall_time();
        timer.restart();

        const double  rhs_norm = rhs.l2_norm();
        SolverControl control(200, 1e-8 * rhs_norm);
        SolverControl control_fast(200, 1e-8 * rhs_norm);

        MyVectorMemory<LinearAlgebra::distributed::BlockVector<double>> memory;
        // try
        //   {
        using SolverType =
          SolverFGMRES<LinearAlgebra::distributed::BlockVector<Number>>;
        typename SolverType::AdditionalData data;
        data.orthogonalization_strategy = LinearAlgebra::
          OrthogonalizationStrategy::delayed_classical_gram_schmidt;
        // data.right_preconditioning = true;
        //  data.exact_residual = false;
        SolverType solver(control_fast, memory, data);

        solver.solve(advection_operator, stage_sol[0], rhs, precondition);
        /*  }
    catch (SolverControl::NoConvergence &)
      {
        typename SolverGMRES<
          LinearAlgebra::distributed::BlockVector<Number>>::AdditionalData
          data;
        data.right_preconditioning      = true;
        data.orthogonalization_strategy = LinearAlgebra::
          OrthogonalizationStrategy::delayed_classical_gram_schmidt;
        data.max_n_tmp_vectors = 20;
        SolverGMRES<LinearAlgebra::distributed::BlockVector<Number>> solver(
          control, memory, data);

        solver.solve(advection_operator, stage_sol[0], rhs, precondition);
      }
        */
        const double my_time = timer.wall_time();

        advection_operator.update_solution(solution, stage_sol[0]);

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
            pcout << " n iter "
                  << control.last_step() + control_fast.last_step() << " "
                  << rhs_norm << " " << control_fast.initial_value() << " "
                  << control_fast.last_value();
            for (unsigned int s = 0; s < step; ++s)
              pcout << " " << project_sol[s];
            pcout << std::endl;
          }
        if (false)
          {
            pcout << " n iter "
                  << control.last_step() + control_fast.last_step() << " "
                  << rhs_norm << " " << control_fast.initial_value() << " "
                  << control_fast.last_value() << " tsol = " << my_time
                  << "  proj ";
            for (const auto d : project_sol)
              pcout << d << " ";
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

    pcout << "   Average wall clock time per time step: "
          << (prep_time + sol_time) / timestep_number << "s, time per element: "
          << (prep_time + sol_time) / timestep_number /
               triangulation->n_global_active_cells()
          << "s" << std::endl;

    pcout << "   Spent " << output_time << "s on output, " << prep_time
          << "s on projection and " << sol_time << "s on solving." << std::endl;

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
