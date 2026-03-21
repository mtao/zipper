#include <zipper/transform/common.hpp>

#include "../catch_include.hpp"

using namespace zipper;
using namespace zipper::transform;

TEST_CASE("radians and degrees", "[transform][common]") {
    CHECK(radians(180.0f) == Catch::Approx(std::numbers::pi_v<float>));
    CHECK(radians(90.0f) == Catch::Approx(std::numbers::pi_v<float> / 2.0f));
    CHECK(radians(0.0f) == Catch::Approx(0.0f));
    CHECK(radians(360.0f) == Catch::Approx(2.0f * std::numbers::pi_v<float>));

    CHECK(degrees(std::numbers::pi_v<float>) == Catch::Approx(180.0f));
    CHECK(degrees(std::numbers::pi_v<float> / 2.0f) == Catch::Approx(90.0f));
    CHECK(degrees(0.0f) == Catch::Approx(0.0f));
}

TEST_CASE("radians/degrees roundtrip", "[transform][common]") {
    CHECK(degrees(radians(45.0)) == Catch::Approx(45.0));
    CHECK(radians(degrees(1.0)) == Catch::Approx(1.0));
}
