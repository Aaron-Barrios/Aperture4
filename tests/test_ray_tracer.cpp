#include "catch2/catch_all.hpp"
#include <catch2/catch_approx.hpp>

#include "systems/ray_tracer_math.hpp"

using namespace Aperture;

TEST_CASE("ray tracer emissivity closure", "[ray_tracer]") {
  SECTION("zero velocity reduces to density scaling") {
    double doppler = ray_tracer_doppler_from_velocity(0.0, 0.0, 0.0);
    REQUIRE(doppler == Catch::Approx(1.0));

    double j = ray_tracer_emissivity(4.0, doppler, 2.5, 3.0);
    REQUIRE(j == Catch::Approx(10.0));
  }

  SECTION("forward motion brightens when beam weight is positive") {
    double doppler = ray_tracer_doppler_from_velocity(0.4, 0.0, 0.0);
    double j0 = ray_tracer_emissivity(4.0, doppler, 1.0, 0.0);
    double j1 = ray_tracer_emissivity(4.0, doppler, 1.0, 3.0);
    REQUIRE(j0 == Catch::Approx(4.0));
    REQUIRE(j1 > j0);
  }

  SECTION("flux to velocity protects against zero density") {
    double v = ray_tracer_flux_to_velocity(0.0, 5.0, 1.0e-12);
    REQUIRE(v == Catch::Approx(5.0e12));
  }
}
