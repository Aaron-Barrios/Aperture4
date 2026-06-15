#pragma once

#include <algorithm>
#include <cmath>

namespace Aperture {

template <typename T>
inline T ray_tracer_doppler_from_velocity(T vx, T vy, T vz) {
  T beta2 = vx * vx + vy * vy + vz * vz;
  beta2 = std::min(beta2, static_cast<T>(0.999999));
  T gamma = static_cast<T>(1) / std::sqrt(static_cast<T>(1) - beta2);
  // Observer at +z looking back along -z:
  // line-of-sight velocity = -v_z, Doppler D = 1/(γ(1 - v_los)) = 1/(γ(1 + v_z))
  return static_cast<T>(1) / (gamma * (static_cast<T>(1) + vz));
}

template <typename T>
inline T ray_tracer_emissivity(T density, T doppler, T base_intensity,
                               T beam_weight) {
  density = std::max(density, static_cast<T>(0));
  return base_intensity * density *
         (static_cast<T>(1) +
          beam_weight * std::max(static_cast<T>(0), doppler - static_cast<T>(1)));
}

template <typename T>
inline T ray_tracer_flux_to_velocity(T density, T flux_component, T eps) {
  return flux_component / std::max(density, eps);
}

}  // namespace Aperture
