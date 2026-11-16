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

TEMPLATE_TEST_CASE("clip_depth_conversion", "[transform][projection]", float, double) {
    using T = TestType;
    const T near = T(1);
    const T far = T(10);
    const auto perspective_projection = perspective(T(1), T(2), near, far);
    const auto orthographic_projection = ortho(T(-2), T(2), T(-1), T(1), near, far);
    const auto offcenter_projection = frustum(T(-1), T(2), T(-1), T(1), near, far);
    const auto ndc_depth = [](const auto &projection, T z) {
        const Vector<T, 4> clip = projection * Vector<T, 4>{T(0), T(0), z, T(1)};
        return clip(2) / clip(3);
    };

    for (const auto &projection : {perspective_projection, orthographic_projection,
                                    offcenter_projection}) {
        const auto converted = clip_depth_zero_to_one(projection);
        CHECK(ndc_depth(converted, -near) == Catch::Approx(T(0)).margin(1e-5));
        CHECK(ndc_depth(converted, -far) == Catch::Approx(T(1)).margin(1e-5));
        const auto restored = clip_depth_minus_one_to_one(converted);
        for (index_type row = 0; row < 4; ++row) {
            for (index_type col = 0; col < 4; ++col) {
                CHECK(restored(row, col) == Catch::Approx(projection(row, col)).margin(1e-5));
                if (row != 2) CHECK(converted(row, col) == projection(row, col));
            }
        }
    }

    const auto infinite = clip_depth_zero_to_one(infinite_perspective(T(1), T(1), near));
    CHECK(ndc_depth(infinite, -near) == Catch::Approx(T(0)).margin(1e-5));
    CHECK(ndc_depth(infinite, T(-1e6)) == Catch::Approx(T(1)).margin(1e-5));
}
