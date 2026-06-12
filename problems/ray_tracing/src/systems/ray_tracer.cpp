#include "framework/config.h"
#include "systems/policies/exec_policy_host.hpp"
#include "ray_tracer_impl.hpp"

namespace Aperture {

template class ray_tracer<Config<2>, exec_policy_host>;

}  // namespace Aperture
