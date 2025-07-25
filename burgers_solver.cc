// This file is part of the advection_miniapp repository and subject to the
// LGPL license. See the LICENSE file in the top level directory of this
// project for details.

// Program for explicit time integration of the multi-dimensional Burgers
// equation Author: Martin Kronbichler, University of Augsburg, 2023
//
// This program has similarities with the step-67 tutorial program of deal.II,
// see https://dealii.org/developer/doxygen/deal.II/step_67.html , but it
// implements a simpler equation and is therefore ideal for learning about
// matrix-free evaluators.

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
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



namespace DGBurgers
{
  using namespace dealii;

  // The dimension can be selected to be 1, 2, 3 (it is a C++ template
  // argument, so different code gets compiled in 1D/2D/3D)
  const unsigned int dimension = 2;

  // This parameter controls the mesh size by the number the initial mesh
  // (consisting of a single line/square/cube) is refined by doubling the
  // number of elements for every increase in number. Thus, the number of
  // elements is given by 2^(dim * n_global_refinements)
  const unsigned int n_min_global_refinements = 5;
  const unsigned int n_max_global_refinements = 5;

  // The time step size is controlled via this parameter as
  // dt = courant_number * min_h / (transport_norm * fe_degree^1.5)
  const double courant_number = 0.05;

  // 0: central flux, 1: classical upwind flux (= Lax-Friedrichs)
  const double flux_alpha = 1.0;

  // The final simulation time
  const double FINAL_TIME = 2.0;

  // Frequency of output
  const double output_tick = 0.05;

  enum LowStorageRungeKuttaScheme
  {
    stage_3_order_3, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_5_order_4, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_7_order_4, /* Tselios, Simos, 2007 */
    stage_9_order_5, /* Kennedy, Carpenter, Lewis, 2000 */
  };
  constexpr LowStorageRungeKuttaScheme lsrk_scheme = stage_5_order_4;

  // Enable or disable writing of result files for visualization with ParaView
  // or VisIt
  const bool print_vtu = true;


  // Analytical solution of the problem
  template <int dim>
  class InitialCondition : public Function<dim>
  {
  public:
    InitialCondition()
      : Function<dim>(dim)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    template <typename Number>
    Number
    value(const Point<dim, Number> &p, const unsigned int component = 0) const
    {
      if (component == 0)
        return -std::sin(numbers::PI * p[0]);
      else
        return 0.;
    }
  };



  // Implementation of the Burgers operation
  template <int dim>
  class BurgersOperation
  {
  public:
    typedef double Number;

    BurgersOperation()
      : computing_times(3)
    {}

    void
    reinit(const Mapping<dim> &mapping, const DoFHandler<dim> &dof_handler);

    void
    initialize_dof_vector(LinearAlgebra::distributed::Vector<Number> &vec)
    {
      data.initialize_dof_vector(vec);
    }

    ~BurgersOperation()
    {
      if (computing_times[2] > 0)
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "Burgers operator evaluated "
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
          LinearAlgebra::distributed::Vector<Number>       &dst,
          const double                                      current_time);

    void
    perform_stage(const Number current_time,
                  const Number factor_solution,
                  const Number factor_ai,
                  const LinearAlgebra::distributed::Vector<Number> &current_ri,
                  LinearAlgebra::distributed::Vector<Number>       &vec_ki,
                  LinearAlgebra::distributed::Vector<Number>       &solution,
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
                      LinearAlgebra::distributed::Vector<Number>       &dst);

    void
    local_apply_inverse_mass_matrix(
      const MatrixFree<dim, Number>                    &data,
      LinearAlgebra::distributed::Vector<Number>       &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int>      &cell_range) const;

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
  };



  template <int dim>
  void
  BurgersOperation<dim>::reinit(const Mapping<dim>    &mapping,
                                const DoFHandler<dim> &dof_handler)
  {
    const unsigned int fe_degree  = dof_handler.get_fe().degree;
    Quadrature<1>      quadrature = QGauss<1>(fe_degree * (fe_degree + 1) / 2);
    Quadrature<1>      quadrature_mass = QGauss<1>(fe_degree + 1);
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
  }



  template <int dim>
  void
  BurgersOperation<dim>::local_apply_domain(
    const MatrixFree<dim, Number>                    &data,
    LinearAlgebra::distributed::Vector<Number>       &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int>      &cell_range) const
  {
    FEEvaluation<dim, -1, 0, dim, Number> eval(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);

        // compute u^h(x) from src
        eval.gather_evaluate(src, EvaluationFlags::values);

        // loop over quadrature points and compute the local volume flux
        for (const unsigned int q : eval.quadrature_point_indices())
          {
            const auto u = eval.get_value(q);
            eval.submit_gradient(outer_product(u, u), q);
          }

        // multiply by nabla v^h(x) and sum
        eval.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }



  template <int dim>
  void
  BurgersOperation<dim>::local_apply_inner_face(
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
    FEFaceEvaluation<dim, -1, 0, dim, Number> eval_minus(data, true);
    FEFaceEvaluation<dim, -1, 0, dim, Number> eval_plus(data, false);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_minus.gather_evaluate(src, EvaluationFlags::values);
        eval_plus.reinit(face);
        eval_plus.gather_evaluate(src, EvaluationFlags::values);

        for (const unsigned int q : eval_minus.quadrature_point_indices())
          {
            const auto u_minus       = eval_minus.get_value(q);
            const auto u_plus        = eval_plus.get_value(q);
            const auto normal_vector = eval_minus.normal_vector(q);

            const VectorizedArray<Number> u_minus_n = u_minus * normal_vector;
            const VectorizedArray<Number> u_plus_n  = u_plus * normal_vector;
            const auto                    flux_times_normal_of_minus =
              0.5 *
              ((u_minus_n * u_minus + u_plus_n * u_plus) +
               flux_alpha * std::max(std::abs(u_minus_n), std::abs(u_plus_n)) *
                 (u_minus - u_plus));

            // We want to test 'flux_times_normal' by the test function, which
            // is called 'FEEvaluation::submit_value(). We need a minus sign
            // for the minus side (interior face) because the boundary term is
            // located on the right hand side and should get a minus sign. On
            // the exterior/plus side, the normal vector has the opposite
            // sign. Instead of recomputing the flux times the normal vector
            // of the plus side, we simply switch the sign here
            eval_minus.submit_value(-flux_times_normal_of_minus, q);
            eval_plus.submit_value(flux_times_normal_of_minus, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values, dst);
        eval_plus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  template <int dim>
  void
  BurgersOperation<dim>::local_apply_boundary_face(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &,
    const LinearAlgebra::distributed::Vector<Number> &,
    const std::pair<unsigned int, unsigned int> &) const
  {
    AssertThrow(false, ExcNotImplemented());
  }



  template <int dim>
  void
  BurgersOperation<dim>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, Number>                    &data,
    LinearAlgebra::distributed::Vector<Number>       &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int>      &cell_range) const
  {
    FEEvaluation<dim, -1, 0, dim, Number> eval(data, 0, 1);

    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim, Number>
      inverse(eval);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval.read_dof_values(src);

        inverse.apply(eval.begin_dof_values(), eval.begin_dof_values());

        eval.set_dof_values(dst);
      }
  }



  template <int dim>
  void
  BurgersOperation<dim>::apply(
    const LinearAlgebra::distributed::Vector<Number> &src,
    LinearAlgebra::distributed::Vector<Number>       &dst,
    const double                                      current_time)
  {
    time = current_time;

    Timer timer;
    data.loop(&BurgersOperation::local_apply_domain,
              &BurgersOperation::local_apply_inner_face,
              &BurgersOperation::local_apply_boundary_face,
              this,
              dst,
              src,
              true,
              MatrixFree<dim, Number>::DataAccessOnFaces::values,
              MatrixFree<dim, Number>::DataAccessOnFaces::values);
    computing_times[0] += timer.wall_time();

    timer.restart();
    data.cell_loop(&BurgersOperation::local_apply_inverse_mass_matrix,
                   this,
                   dst,
                   dst);
    computing_times[1] += timer.wall_time();

    computing_times[2] += 1.;
  }



  template <int dim>
  void
  BurgersOperation<dim>::perform_stage(
    const Number                                      current_time,
    const Number                                      factor_solution,
    const Number                                      factor_ai,
    const LinearAlgebra::distributed::Vector<Number> &current_ri,
    LinearAlgebra::distributed::Vector<Number>       &vec_ki,
    LinearAlgebra::distributed::Vector<Number>       &solution,
    LinearAlgebra::distributed::Vector<Number>       &next_ri) const
  {
    time = current_time;

    Timer timer;

    data.loop(&BurgersOperation::local_apply_domain,
              &BurgersOperation::local_apply_inner_face,
              &BurgersOperation::local_apply_boundary_face,
              this,
              vec_ki,
              current_ri,
              true,
              MatrixFree<dim, Number>::DataAccessOnFaces::values,
              MatrixFree<dim, Number>::DataAccessOnFaces::values);
    computing_times[0] += timer.wall_time();

    timer.restart();
    data.cell_loop(
      &BurgersOperation::local_apply_inverse_mass_matrix,
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



  template <int dim>
  void
  BurgersOperation<dim>::project_initial(
    LinearAlgebra::distributed::Vector<Number> &dst) const
  {
    InitialCondition<dim>                 initial_condition;
    FEEvaluation<dim, -1, 0, dim, Number> phi(data, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim, Number>
      inverse(phi);
    dst.zero_out_ghost_values();
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        for (const unsigned int q : phi.quadrature_point_indices())
          {
            Tensor<1, dim, VectorizedArray<Number>> value;
            for (unsigned int d = 0; d < dim; ++d)
              value[d] = initial_condition.value(phi.quadrature_point(q), d);
            phi.submit_dof_value(value, q);
          }
        inverse.transform_from_q_points_to_basis(dim,
                                                 phi.begin_dof_values(),
                                                 phi.begin_dof_values());
        phi.set_dof_values(dst);
      }
  }


  template <int dim>
  Tensor<1, 3>
  BurgersOperation<dim>::compute_mass_and_energy(
    const LinearAlgebra::distributed::Vector<Number> &vec) const
  {
    Tensor<1, 3>                          mass_energy = {};
    FEEvaluation<dim, -1, 0, dim, Number> phi(data);
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
            mass += phi.get_value(q)[0] * phi.JxW(q);
            energy += phi.get_value(q) * phi.get_value(q) * phi.JxW(q);
            H1semi += scalar_product(phi.get_gradient(q), phi.get_gradient(q)) *
                      phi.JxW(q);
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
                      VectorType     &solution,
                      VectorType     &vec_ri,
                      VectorType     &vec_ki) const
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
  class BurgersProblem
  {
  public:
    typedef typename BurgersOperation<dim>::Number Number;

    BurgersProblem(const unsigned int fe_degree);

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
    MappingQ<dim>                       mapping;
    FESystem<dim>                       fe;
    DoFHandler<dim>                     dof_handler;

    IndexSet locally_relevant_dofs;

    double time, time_step;

    ConditionalOStream pcout;
  };



  template <int dim>
  BurgersProblem<dim>::BurgersProblem(const unsigned int fe_degree)
    : mapping(fe_degree)
    , fe(FE_DGQ<dim>(fe_degree), dim)
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
  BurgersProblem<dim>::make_grid(const unsigned int n_refinements)
  {
    time      = 0;
    time_step = 0;
    triangulation->clear();

    GridGenerator::hyper_cube(*triangulation, -1, 1);
    for (const auto &cell : triangulation->cell_iterators())
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        if (cell->at_boundary(f))
          cell->face(f)->set_all_boundary_ids(f);
    std::vector<
      GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
      periodic_faces;
    for (unsigned int d = 0; d < dim; ++d)
      GridTools::collect_periodic_faces(
        *triangulation, 2 * d, 2 * d + 1, d, periodic_faces);
    triangulation->add_periodicity(periodic_faces);

    triangulation->refine_global(n_refinements);

    pcout << "   Number of elements:            "
          << triangulation->n_global_active_cells() << std::endl;
  }



  template <int dim>
  void
  BurgersProblem<dim>::setup_dofs()
  {
    dof_handler.reinit(*triangulation);
    dof_handler.distribute_dofs(fe);

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

    time_step =
      courant_number * glob_min_vertex_distance /
      std::pow(static_cast<double>(std::max(1U, dof_handler.get_fe().degree)),
               1.5);

    time_step = FINAL_TIME / std::ceil(FINAL_TIME / time_step);

    if (time == 0)
      pcout << "   Time step size: " << time_step
            << ", minimum vertex distance: " << glob_min_vertex_distance
            << std::endl
            << std::endl;
  }



  template <int dim>
  void
  BurgersProblem<dim>::output_results(const unsigned int output_number,
                                      const Tensor<1, 3> mass_energy)
  {
    Vector<double> norm_per_cell(triangulation->n_active_cells());

    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      Functions::ZeroFunction<dim>(dim),
                                      norm_per_cell,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::L2_norm);
    const double magnitude =
      VectorTools::compute_global_error(*triangulation,
                                        norm_per_cell,
                                        VectorTools::L2_norm);
    pcout << "   Time" << std::setw(8) << std::setprecision(3) << time
          << "  magnitude " << std::setprecision(5) << std::setw(10)
          << magnitude << "  mass " << std::setprecision(10) << std::setw(16)
          << mass_energy[0] << "  energy " << std::setprecision(10)
          << std::setw(16) << mass_energy[1] << "  H1-semi "
          << std::setprecision(4) << std::setw(9) << mass_energy[2]
          << std::endl;

    if (!print_vtu)
      return;

    // Write output to a vtu file
    DataOut<dim>          data_out;
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.add_data_vector(
      dof_handler,
      solution,
      std::vector<std::string>(dim, "solution"),
      std::vector<DataComponentInterpretation::DataComponentInterpretation>(
        dim, DataComponentInterpretation::component_is_part_of_vector));
    data_out.build_patches(mapping,
                           dof_handler.get_fe().degree,
                           DataOut<dim>::curved_inner_cells);

    const std::string filename =
      "solution_" + Utilities::int_to_string(output_number, 3) + ".vtu";
    data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
  }



  template <int dim>
  void
  BurgersProblem<dim>::run(const unsigned int n_refinements)
  {
    make_grid(n_refinements);
    setup_dofs();

    // Initialize the Burgers operator and the time integrator that will
    // perform all interesting steps
    BurgersOperation<dim> burgers_operator;
    burgers_operator.reinit(mapping, dof_handler);
    burgers_operator.initialize_dof_vector(solution);
    burgers_operator.project_initial(solution);

    unsigned int n_output = 0;
    output_results(n_output++,
                   burgers_operator.compute_mass_and_energy(solution));

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

        time_integrator.perform_time_step(burgers_operator,
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
            output_results(n_output++,
                           burgers_operator.compute_mass_and_energy(solution));
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

    // As 'burgers_operator' goes out of scope, it will call its constructor
    // that prints the accumulated computing times over all time steps to
    // screen
  }
} // namespace DGBurgers



int
main(int argc, char **argv)
{
  using namespace DGBurgers;
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  try
    {
      deallog.depth_console(0);

      unsigned int degree = 3;
      if (argc > 1)
        degree = std::atoi(argv[1]);

      // The actual dimension is selected by inserting the global constant
      // 'dimension' as the actual template argument here, rather than the
      // placeholder 'dim' used as *template* in the class definitions above.
      BurgersProblem<dimension> problem(degree);
      for (unsigned int r = n_min_global_refinements;
           r <= n_max_global_refinements;
           ++r)
        problem.run(r);
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
