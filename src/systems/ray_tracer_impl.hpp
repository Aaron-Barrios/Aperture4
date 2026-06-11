#pragma once

#include "ray_tracer.h"
#include "ray_tracer_math.hpp"
#include "framework/environment.h"
#include "utils/logger.h"

namespace Aperture {

template <class Conf, template <class> class ExecPolicy>
ray_tracer<Conf, ExecPolicy>::ray_tracer(const grid_t<Conf>& grid,
                                         const domain_comm<Conf, ExecPolicy>* comm)
    : m_grid(grid), m_comm(comm) {
  // read parameters if present
  sim_env().params().get_value("rt_output_interval", m_output_interval);
  sim_env().params().get_value("rt_base_intensity", m_base_intensity);
  sim_env().params().get_value("rt_beam_weight", m_beam_weight);
}

template <class Conf, template <class> class ExecPolicy>
ray_tracer<Conf, ExecPolicy>::~ray_tracer() = default;

template <class Conf, template <class> class ExecPolicy>
void
ray_tracer<Conf, ExecPolicy>::register_data_components() {
  sim_env().get_data("num_e", m_num_e);
  sim_env().get_data("flux_e", m_flux_e);

  extent_t<2> image_ext(this->m_grid.reduced_dim(0),
                        this->m_grid.reduced_dim(1));
  m_image = sim_env().template register_data<multi_array_data<value_t, 2>>(
      "image", image_ext, ExecPolicy<Conf>::data_mem_type());
  m_image->include_in_snapshot(true);
}

template <class Conf, template <class> class ExecPolicy>
void
ray_tracer<Conf, ExecPolicy>::init() {
  Logger::print_info("ray_tracer initialized (interval={} base={} beam={})",
                     m_output_interval, m_base_intensity, m_beam_weight);
}

//How we update the intensity field
template <class Conf, template <class> class ExecPolicy>
void
ray_tracer<Conf, ExecPolicy>::update(double dt, uint32_t step) {
  if (step % m_output_interval != 0) return;

  if (m_num_e == nullptr || m_flux_e == nullptr || m_image == nullptr) {
    Logger::print_err("ray_tracer: missing required data components");
    return;
  }

  // This tracer uses the already-computed fluid moments as an emissivity map.
  // We integrate along the +x direction and store a brightness value for each
  // pixel (x, y) in a 2D framebuffer.
  m_num_e->copy_to_host();
  m_flux_e->copy_to_host();
  m_image->init();

  auto num_ptr = m_num_e->host_ndptr();
  auto flux_ptr = m_flux_e->host_ptrs();
  auto img_ptr = m_image->host_ndptr();

  auto img_ext = m_image->extent();
  if (img_ext[0] == 0 || img_ext[1] == 0) return;

  constexpr value_t eps = static_cast<value_t>(1.0e-12);
  for (uint32_t iy = 0; iy < img_ext[1]; ++iy) {
    value_t line_integral = 0;
    for (uint32_t ix = 0; ix < img_ext[0]; ++ix) {
      index_t<2> pixel(ix, iy);
      index_t<2> pos(ix + m_grid.guard[0], iy + m_grid.guard[1]);
      auto idx = m_image->get_idx(pixel);

      value_t n = std::max(static_cast<value_t>(0), num_ptr[idx]);
      value_t fx = flux_ptr[0][idx];
      value_t fy = flux_ptr[1][idx];
      value_t fz = flux_ptr[2][idx];

      value_t vx = ray_tracer_flux_to_velocity(n, fx, eps);
      value_t vy = ray_tracer_flux_to_velocity(n, fy, eps);
      value_t vz = ray_tracer_flux_to_velocity(n, fz, eps);

      // Fixed line of sight along +x for this prototype.
      value_t doppler = ray_tracer_doppler_from_velocity(vx, vy, vz);

      // Simple emissivity closure:
      //   j = j0 * n * (1 + beam_weight * max(0, doppler - 1))
      // so density dominates, but bulk motion can brighten the ray.
      value_t emissivity = ray_tracer_emissivity(n, doppler, m_base_intensity,
                                                 m_beam_weight);

      line_integral += emissivity * static_cast<value_t>(m_grid.delta[0]);
      img_ptr[idx] = line_integral;
    }
  }

  m_image->copy_to_device();

  Logger::print_detail("ray_tracer: updated image at step {}", step);
}

}  // namespace Aperture
