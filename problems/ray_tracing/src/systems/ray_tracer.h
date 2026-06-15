#pragma once

#include "data/fields.h"
#include "data/multi_array_data.hpp"
#include "framework/system.h"
#include "systems/domain_comm.h"
#include "systems/grid.h"
#include "systems/policies.h"
#include "utils/nonown_ptr.hpp"
#include <string>

namespace Aperture {

template <class Conf, template <class> class ExecPolicy>
class ray_tracer : public system_t {
 public:
  using value_t = typename Conf::value_t;
  static std::string name() { return "ray_tracer"; }

  ray_tracer(const grid_t<Conf>& grid,
             const domain_comm<Conf, ExecPolicy>* comm = nullptr);

  virtual ~ray_tracer();

  void init() override;
  void register_data_components() override;
  void update(double dt, uint32_t step) override;
  void write_image_pgm(const std::string& filename) const;

 protected:
  const grid_t<Conf>& m_grid;
  const domain_comm<Conf, ExecPolicy>* m_comm = nullptr;

  // associated data components used to build the image
  nonown_ptr<scalar_field<Conf>> m_num_e;
  nonown_ptr<field_t<3, Conf>> m_flux_e;

  // 2D framebuffer written each step
  nonown_ptr<multi_array_data<value_t, 2>> m_image;

  // simple parameters
  uint32_t m_output_interval = 1;
  value_t m_base_intensity = 1.0;
  value_t m_beam_weight = 3.0;
  bool m_write_pgm = false;
  std::string m_pgm_prefix = "rt_image";
  std::string m_output_dir = "bin/";
  std::string m_output_dir = "Data/";
};

}  // namespace Aperture
