/*
 * Ray tracer test problem for Aperture4.
 * Sets up a 3D Cartesian PIC simulation with the ray tracer enabled.
 *
 * The ray tracer integrates fluid moments (num_e + flux_e) along z
 * at each output step, projecting onto the (x,y) image plane as
 * seen by an observer at +z looking back along -z.
 *
 * Build:  cmake .. && make raytrace
 * Run:    ./problems/ray_tracing/bin/raytrace -c config.toml
 */

#include "core/particles_functions.h"
#include "data/fields.h"
#include "data/particle_data.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/compute_moments.h"
#include "systems/field_solver_cartesian.h"
#include "systems/grid.h"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/exec_policy_host.hpp"
#include "systems/ptc_updater.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/ray_tracer.h"
#include "systems/data_exporter.h"
#include "utils/util_functions.h"
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<3> Conf;
  using value_t = Conf::value_t;

  auto &env = sim_environment::instance(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::debug);

  // --- Particle tracking (must be set via add(), not just TOML,
  //     because get_value has no size_t overload) ---
  env.params().add("max_tracked_num", (int64_t)50000);
  env.params().add("ptc_output_interval", (int64_t)5);

  // --- Grid & domain (constructed directly, not via register_system) ---
  domain_comm<Conf, exec_policy_host> comm;
  grid_t<Conf> grid(comm);

  // --- Field solver ---
  auto solver = env.register_system<
      field_solver<Conf, exec_policy_host, coord_policy_cartesian>>(grid, &comm);

  // --- Particle pusher ---
  auto pusher = env.register_system<
      ptc_updater<Conf, exec_policy_host, coord_policy_cartesian>>(grid, &comm);

  // --- Moment computer (produces num_e, flux_e consumed by ray tracer) ---
  auto moments = env.register_system<compute_moments<Conf, exec_policy_host>>(grid);

  // --- Tracked particle gatherer (copies tracked particles to ptc.*.h5) ---
  auto tracker = env.register_system<gather_tracked_ptc<Conf, exec_policy_host>>(grid);

  // --- Ray tracer ---
  auto tracer = env.register_system<ray_tracer<Conf, exec_policy_host>>(
      grid, &comm);

  // --- Data exporter (writes HDF5 snapshots that include the "image" array) ---
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_host>>(
      grid, &comm);

  env.init();

  // --- DEBUG: verify params and system state ---
  {
    int64_t dbg_max_tracked = 0, dbg_ptc_interval = 0;
    env.params().get_value("max_tracked_num", dbg_max_tracked);
    env.params().get_value("ptc_output_interval", dbg_ptc_interval);
    std::cerr << "DEBUG: max_tracked_num=" << dbg_max_tracked
              << " ptc_output_interval=" << dbg_ptc_interval
              << " max_ptc_num=" << env.params().get_as<int64_t>("max_ptc_num", 0)
              << std::endl;
  }

  // ============================================================
  // Initial conditions
  // ============================================================

  // Read injection parameters (defaults are sane for a quick test)
  int64_t num_particles = 10000;
  double particle_energy = 10.0;        // relativistic gamma ~ 10
  double particle_weight = 1.0;
  double center_x = 0.5;                // fractional position in domain
  double center_y = 0.5;
  double center_z = 0.5;
  double spread_x = 0.2;               // fractional spread in domain
  double spread_y = 0.2;
  double spread_z = 0.2;

  env.params().get_value("num_particles", num_particles);
  env.params().get_value("particle_energy", particle_energy);
  env.params().get_value("particle_weight", particle_weight);
  env.params().get_value("center_x", center_x);
  env.params().get_value("center_y", center_y);
  env.params().get_value("center_z", center_z);
  env.params().get_value("spread_x", spread_x);
  env.params().get_value("spread_y", spread_y);
  env.params().get_value("spread_z", spread_z);

  // --- Zero out E and B fields ---
  vector_field<Conf> *E, *B;
  sim_env().get_data("Edelta", &E);
  sim_env().get_data("Bdelta", &B);
  E->set_values(0, [](auto x, auto y, auto z) { return 0.0; });
  E->set_values(1, [](auto x, auto y, auto z) { return 0.0; });
  E->set_values(2, [](auto x, auto y, auto z) { return 0.0; });
  B->set_values(0, [](auto x, auto y, auto z) { return 0.0; });
  B->set_values(1, [](auto x, auto y, auto z) { return 0.0; });
  B->set_values(2, [](auto x, auto y, auto z) { return 0.0; });

  // --- Inject a blob of electrons with random momenta ---
  particle_data_t *ptc;
  sim_env().get_data("particles", &ptc);
  ptc->include_in_snapshot(true);   // so particle x1/x2/x3 appear in fld.*.h5

  double Lx = grid.sizes[0];
  double Ly = grid.sizes[1];
  double Lz = grid.sizes[2];
  double x0 = grid.lower[0] + Lx * center_x;
  double y0 = grid.lower[1] + Ly * center_y;
  double z0 = grid.lower[2] + Lz * center_z;
  double sx = Lx * spread_x;
  double sy = Ly * spread_y;
  double sz = Lz * spread_z;

  srand(42);  // deterministic seed for reproducibility

  for (int64_t i = 0; i < num_particles; ++i) {
    // Random position in a Gaussian-like blob (box-Muller approximation)
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    double u3 = (double)rand() / RAND_MAX;
    double u4 = (double)rand() / RAND_MAX;
    double u5 = (double)rand() / RAND_MAX;
    double u6 = (double)rand() / RAND_MAX;

    // Box-Muller for x, y, and z
    double g1 = sqrt(-2.0 * log(max(u1, 1e-12))) * cos(2.0 * M_PI * u2);
    double g2 = sqrt(-2.0 * log(max(u3, 1e-12))) * cos(2.0 * M_PI * u4);
    double g3 = sqrt(-2.0 * log(max(u5, 1e-12))) * cos(2.0 * M_PI * u6);

    double px = x0 + sx * g1;
    double py = y0 + sy * g2;
    double pz = z0 + sz * g3;

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
        {static_cast<value_t>(px),
         static_cast<value_t>(py),
         static_cast<value_t>(pz)},  // position (global, 3D)
        {static_cast<value_t>(mom_x),
         static_cast<value_t>(mom_y),
         static_cast<value_t>(mom_z)},  // momentum
        static_cast<value_t>(particle_weight),
        set_ptc_type_flag(flag_or(PtcFlag::tracked), type));
  }

  Logger::print_info("Injected {} particles (E = {:.1f})", num_particles,
                     particle_energy);

  // --- DEBUG: verify first particle's flag ---
  {
    auto& flags = ptc->flag;
    uint32_t flag0 = flags[0];
    std::cerr << "DEBUG: Injected " << ptc->number() << " particles. "
              << "First particle flag=0x" << std::hex << flag0 << std::dec
              << " has_tracked=" << check_flag(flag0, PtcFlag::tracked)
              << " type=" << (flag0 >> 28)
              << " cell=" << ptc->cell[0]
              << " x1=" << ptc->x1[0]
              << std::endl;
  }

  env.run();
  return 0;
}
