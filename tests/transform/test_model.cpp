#include <cmath>
#include <numbers>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/transform/model.hpp>
#include <zipper/transform/common.hpp>

#include "../catch_include.hpp"

using namespace zipper;
using namespace zipper::transform;

TEST_CASE("translate creates translation transform", "[transform][model]") {
    Vector<float, 3> v({1.0f, 2.0f, 3.0f});
    auto T = translate(v);

    // Translation column should be the translation vector
    CHECK(T(0, 3) == Catch::Approx(1.0f));
    CHECK(T(1, 3) == Catch::Approx(2.0f));
    CHECK(T(2, 3) == Catch::Approx(3.0f));
    CHECK(T(3, 3) == Catch::Approx(1.0f));

    // Diagonal should be 1 (identity linear block)
    CHECK(T(0, 0) == Catch::Approx(1.0f));
    CHECK(T(1, 1) == Catch::Approx(1.0f));
    CHECK(T(2, 2) == Catch::Approx(1.0f));
}

TEST_CASE("translate applied to point via composition", "[transform][model]") {
    Vector<float, 3> v({5.0f, 10.0f, 15.0f});
    auto T = translate(v);

    // Apply to a point (D-dimensional affine action)
    Vector<float, 3> point({1.0f, 2.0f, 3.0f});
    Vector<float, 3> result = T * point;

    CHECK(result(0) == Catch::Approx(6.0f));
    CHECK(result(1) == Catch::Approx(12.0f));
    CHECK(result(2) == Catch::Approx(18.0f));
}

TEST_CASE("rotate 90 degrees around Z", "[transform][model]") {
    Vector<float, 3> zAxis({0.0f, 0.0f, 1.0f});
    auto R = rotate(radians(90.0f), zAxis);

    // Rotating (1, 0, 0) by 90 around Z should give (0, 1, 0)
    Vector<float, 3> point({1.0f, 0.0f, 0.0f});
    Vector<float, 3> result = R * point;

    CHECK(result(0) == Catch::Approx(0.0f).margin(1e-6f));
    CHECK(result(1) == Catch::Approx(1.0f).margin(1e-6f));
    CHECK(result(2) == Catch::Approx(0.0f).margin(1e-6f));
}

TEST_CASE("rotate 360 degrees is identity", "[transform][model]") {
    Vector<float, 3> axis({0.0f, 1.0f, 0.0f});
    auto R = rotate(radians(360.0f), axis);

    for (index_type r = 0; r < 4; ++r) {
        for (index_type c = 0; c < 4; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(R(r, c) == Catch::Approx(expected).margin(1e-5f));
        }
    }
}

TEST_CASE("scale uniform", "[transform][model]") {
    Vector<float, 3> s({2.0f, 2.0f, 2.0f});
    auto S = scale(s);

    Vector<float, 3> point({1.0f, 1.0f, 1.0f});
    Vector<float, 3> result = S * point;

    CHECK(result(0) == Catch::Approx(2.0f));
    CHECK(result(1) == Catch::Approx(2.0f));
    CHECK(result(2) == Catch::Approx(2.0f));
}

TEST_CASE("scale non-uniform", "[transform][model]") {
    Vector<float, 3> s({1.0f, 2.0f, 3.0f});
    auto S = scale(s);

    CHECK(S(0, 0) == Catch::Approx(1.0f));
    CHECK(S(1, 1) == Catch::Approx(2.0f));
    CHECK(S(2, 2) == Catch::Approx(3.0f));
    CHECK(S(3, 3) == Catch::Approx(1.0f));
}

TEST_CASE("translate then rotate via composition", "[transform][model]") {
    Vector<float, 3> v({1.0f, 0.0f, 0.0f});
    Vector<float, 3> zAxis({0.0f, 0.0f, 1.0f});

    auto T = translate(v);
    auto R = rotate(radians(90.0f), zAxis);

    // R * T: first translate, then rotate
    auto RT = R * T;

    // The origin (0,0,0) should be translated to (1,0,0) then rotated to (0,1,0)
    Vector<float, 3> origin({0.0f, 0.0f, 0.0f});
    Vector<float, 3> result = RT * origin;

    CHECK(result(0) == Catch::Approx(0.0f).margin(1e-6f));
    CHECK(result(1) == Catch::Approx(1.0f).margin(1e-6f));
    CHECK(result(2) == Catch::Approx(0.0f).margin(1e-6f));
}

TEST_CASE("translate 2D", "[transform][model][2d]") {
    Vector<float, 2> v({3.0f, 4.0f});
    auto T = translate(v);

    Vector<float, 2> point({1.0f, 2.0f});
    Vector<float, 2> result = T * point;

    CHECK(result(0) == Catch::Approx(4.0f));
    CHECK(result(1) == Catch::Approx(6.0f));
}

TEST_CASE("scale 2D", "[transform][model][2d]") {
    Vector<float, 2> s({2.0f, 3.0f});
    auto S = scale(s);

    Vector<float, 2> point({1.0f, 1.0f});
    Vector<float, 2> result = S * point;

    CHECK(result(0) == Catch::Approx(2.0f));
    CHECK(result(1) == Catch::Approx(3.0f));
}
