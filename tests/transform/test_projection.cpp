#include <cmath>
#include <numbers>

#include <zipper/Matrix.hpp>
#include <zipper/transform/projection.hpp>
#include <zipper/transform/common.hpp>

#include "../catch_include.hpp"

using namespace zipper;
using namespace zipper::transform;

TEST_CASE("ortho 2D", "[transform][projection]") {
    auto P = ortho(-1.0f, 1.0f, -1.0f, 1.0f);
    // Should map [-1,1] to [-1,1] in x and y
    CHECK(P(0, 0) == Catch::Approx(1.0f));
    CHECK(P(1, 1) == Catch::Approx(1.0f));
    CHECK(P(2, 2) == Catch::Approx(-1.0f));
    CHECK(P(3, 3) == Catch::Approx(1.0f));
    // Translation should be zero for symmetric bounds
    CHECK(P(0, 3) == Catch::Approx(0.0f));
    CHECK(P(1, 3) == Catch::Approx(0.0f));
}

TEST_CASE("ortho 3D", "[transform][projection]") {
    auto P = ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);
    CHECK(P(0, 0) == Catch::Approx(1.0f));
    CHECK(P(1, 1) == Catch::Approx(1.0f));
    CHECK(P(3, 3) == Catch::Approx(1.0f));
    // Z scale
    CHECK(P(2, 2) == Catch::Approx(-2.0f / 99.9f));
}

TEST_CASE("perspective", "[transform][projection]") {
    float fovy = radians(45.0f);
    auto P = perspective(fovy, 16.0f / 9.0f, 0.1f, 100.0f);

    float tanHalf = std::tan(fovy / 2.0f);
    CHECK(P(0, 0) == Catch::Approx(1.0f / (16.0f / 9.0f * tanHalf)));
    CHECK(P(1, 1) == Catch::Approx(1.0f / tanHalf));
    CHECK(P(3, 2) == Catch::Approx(-1.0f));
    // w column should be 0 except (3,2)
    CHECK(P(0, 3) == Catch::Approx(0.0f));
    CHECK(P(1, 3) == Catch::Approx(0.0f));
    CHECK(P(3, 3) == Catch::Approx(0.0f));
}

TEST_CASE("frustum symmetric equals perspective", "[transform][projection]") {
    float fovy = radians(60.0f);
    float aspect = 1.5f;
    float near = 0.1f;
    float far = 50.0f;

    auto P = perspective(fovy, aspect, near, far);

    float tanHalf = std::tan(fovy / 2.0f);
    float top = near * tanHalf;
    float right = top * aspect;
    auto F = frustum(-right, right, -top, top, near, far);

    for (index_type r = 0; r < 4; ++r) {
        for (index_type c = 0; c < 4; ++c) {
            CHECK(P(r, c) == Catch::Approx(F(r, c)).margin(1e-6f));
        }
    }
}

TEST_CASE("infinite_perspective", "[transform][projection]") {
    float fovy = radians(45.0f);
    auto P = infinite_perspective(fovy, 1.0f, 0.1f);
    CHECK(P(2, 2) == Catch::Approx(-1.0f));
    CHECK(P(2, 3) == Catch::Approx(-0.2f));
    CHECK(P(3, 2) == Catch::Approx(-1.0f));
}
