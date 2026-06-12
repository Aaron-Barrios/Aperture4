/*
 * Ray tracer test problem for Aperture4.
 * Sets up a 2D Cartesian PIC simulation with the ray tracer enabled.
 *
 * The ray tracer integrates fluid moments (num_e + flux_e) along +x
 * at each output step, writing PGM images to disk.
 */

#include "core/particles_functions.h"
#include "data/fields.h"
#include "data/particle_data.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/compute_moments.h"
#include "systems/field_solver_default.h"
#include "systems/grid.h"
#include "systems/policies/exec_policy_host.hpp"
#include "systems/ptc_updater.h"
#include "systems/ray_tracer.h"
#include "systems/data_exporter.h"
#include "utils/util_functions.h"
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  sim_environment env(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::debug);

  // --- Grid & domain ---
  auto comm = env.register_system<domain_comm<Conf>>(env);
  auto grid_sys = env.register_system<grid_t<Conf>>(env, *comm);
  auto& grid = *grid_sys;

  // --- Field solver ---
  auto solver =
      env.register_system<field_solver_default<Conf>>(env, grid, comm);

  // --- Particle pusher ---
  auto pusher = env.register_system<ptc_updater<Conf>>(env, grid, &comm);

  // --- Moment computer (produces num_e, flux_e consumed by ray tracer) ---
  auto moments = env.register_system<compute_moments<Conf, exec_policy_host>>(
      grid);

  // --- Ray tracer ---
  auto tracer = env.register_system<ray_tracer<Conf, exec_policy_host>>(
      grid, &comm);

  // --- Data exporter (writes HDF5 snapshots that include the "image" array) ---
  auto exporter = env.register_system<data_exporter<Conf>>(env, grid, comm);

  env.init();

  // ============================================================
  // Initial conditions
  // ============================================================

  // Read injection parameters (defaults are sane for a quick test)
  int64_t num_particles = 10000;
  double particle_energy = 10.0;        // relativistic gamma ~ 10
  double particle_weight = 1.0;
  double center_x = 0.5;                // fractional position in domain
  double center_y = 0.5;
  double spread_x = 0.2;               // fractional spread in domain
  double spread_y = 0.2;

  env.params().get_value("num_particles", num_particles);
  env.params().get_value("particle_energy", particle_energy);
  env.params().get_value("particle_weight", particle_weight);
  env.params().get_value("center_x", center_x);
  env.params().get_value("center_y", center_y);
  env.params().get_value("spread_x", spread_x);
  env.params().get_value("spread_y", spread_y);

  // --- Zero out E and B fields ---
  vector_field<Conf> *E, *B;
  sim_env().get_data("Edelta", &E);
  sim_env().get_data("Bdelta", &B);
  E->set_to_zero();
  B->set_to_zero();

  // --- Inject a blob of electrons with random momenta ---
  particle_data_t *ptc;
  sim_env().get_data("particles", &ptc);

  double Lx = grid.sizes[0];
  double Ly = grid.sizes[1];
  double x0 = grid.lower[0] + Lx * center_x;
  double y0 = grid.lower[1] + Ly * center_y;
  double sx = Lx * spread_x;
  double sy = Ly * spread_y;

  srand(42);  // deterministic seed for reproducibility

  for (int64_t i = 0; i < num_particles; ++i) {
    // Random position in a Gaussian-like blob (box-Muller approximation)
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    double u3 = (double)rand() / RAND_MAX;
    double u4 = (double)rand() / RAND_MAX;

    // Box-Muller for x and y
    double g1 = sqrt(-2.0 * log(max(u1, 1e-12))) * cos(2.0 * M_PI * u2);
    double g2 = sqrt(-2.0 * log(max(u3, 1e-12))) * cos(2.0 * M_PI * u4);

    double px = x0 + sx * g1;
    double py = y0 + sy * g2;

    // Random velocity on a sphere of constant |p| = particle_energy
    double phi = 2.0 * M_PI * (double)rand() / RAND_MAX;
    double cos_theta = 2.0 * (double)rand() / RAND_MAX - 1.0;
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    double mom_mag = particle_energy;
    double mom_x = mom_mag * sin_theta * cos(phi);
    double mom_y = mom_mag * sin_theta * sin(phi);
    double mom_z = mom_mag * cos_theta;

    // Alternate electrons and positrons
    PtcType type = (i % 2 == 0) ? PtcType::electron : PtcType::positron;

    ptc_append_global(
        exec_tags::host{}, *ptc, grid,
        {static_cast<Conf::value_t>(px),
         static_cast<Conf::value_t>(py),
         0.0},  // position (global)
        {static_cast<Conf::value_t>(mom_x),
         static_cast<Conf::value_t>(mom_y),
         static_cast<Conf::value_t>(mom_z)},  // momentum
        static_cast<Conf::value_t>(particle_weight),
        set_ptc_type_flag(0, type));
  }

  Logger::print_info("Injected {} particles (E = {:.1f})", num_particles,
                     particle_energy);

  env.run();
  return 0;
}
