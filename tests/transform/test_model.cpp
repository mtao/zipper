#include <cmath>
#include <numbers>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/transform/model.hpp>
#include <zipper/transform/common.hpp>

#include "../catch_include.hpp"

using namespace zipper;
using namespace zipper::transform;

TEST_CASE("translate identity", "[transform][model]") {
    AffineTransform<float> I;
    Vector<float, 3> v({1.0f, 2.0f, 3.0f});

    auto result = translate(I, v);

    // Translation column should be the translation vector
    CHECK(result(0, 3) == Catch::Approx(1.0f));
    CHECK(result(1, 3) == Catch::Approx(2.0f));
    CHECK(result(2, 3) == Catch::Approx(3.0f));
    CHECK(result(3, 3) == Catch::Approx(1.0f));

    // Diagonal should be 1
    CHECK(result(0, 0) == Catch::Approx(1.0f));
    CHECK(result(1, 1) == Catch::Approx(1.0f));
    CHECK(result(2, 2) == Catch::Approx(1.0f));
}

TEST_CASE("translate point", "[transform][model]") {
    AffineTransform<float> I;
    Vector<float, 3> v({5.0f, 10.0f, 15.0f});
    auto T = translate(I, v);

    // Apply to a point (w=1)
    Vector<float, 4> point({1.0f, 2.0f, 3.0f, 1.0f});
    Vector<float, 4> result = Vector<float, 4>(T * point);

    CHECK(result(0) == Catch::Approx(6.0f));
    CHECK(result(1) == Catch::Approx(12.0f));
    CHECK(result(2) == Catch::Approx(18.0f));
    CHECK(result(3) == Catch::Approx(1.0f));
}

TEST_CASE("rotate 90 degrees around Z", "[transform][model]") {
    AffineTransform<float> I;
    Vector<float, 3> zAxis({0.0f, 0.0f, 1.0f});
    auto R = rotate(I, radians(90.0f), zAxis);

    // Rotating (1, 0, 0, 1) by 90 around Z should give (0, 1, 0, 1)
    Vector<float, 4> point({1.0f, 0.0f, 0.0f, 1.0f});
    Vector<float, 4> result = Vector<float, 4>(R * point);

    CHECK(result(0) == Catch::Approx(0.0f).margin(1e-6f));
    CHECK(result(1) == Catch::Approx(1.0f).margin(1e-6f));
    CHECK(result(2) == Catch::Approx(0.0f).margin(1e-6f));
    CHECK(result(3) == Catch::Approx(1.0f).margin(1e-6f));
}

TEST_CASE("rotate 360 degrees is identity", "[transform][model]") {
    AffineTransform<float> I;
    Vector<float, 3> axis({0.0f, 1.0f, 0.0f});
    auto R = rotate(I, radians(360.0f), axis);

    for (index_type r = 0; r < 4; ++r) {
        for (index_type c = 0; c < 4; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(R(r, c) == Catch::Approx(expected).margin(1e-5f));
        }
    }
}

TEST_CASE("scale uniform", "[transform][model]") {
    AffineTransform<float> I;
    Vector<float, 3> s({2.0f, 2.0f, 2.0f});
    auto S = scale(I, s);

    Vector<float, 4> point({1.0f, 1.0f, 1.0f, 1.0f});
    Vector<float, 4> result = Vector<float, 4>(S * point);

    CHECK(result(0) == Catch::Approx(2.0f));
    CHECK(result(1) == Catch::Approx(2.0f));
    CHECK(result(2) == Catch::Approx(2.0f));
    CHECK(result(3) == Catch::Approx(1.0f));
}

TEST_CASE("scale non-uniform", "[transform][model]") {
    AffineTransform<float> I;
    Vector<float, 3> s({1.0f, 2.0f, 3.0f});
    auto S = scale(I, s);

    CHECK(S(0, 0) == Catch::Approx(1.0f));
    CHECK(S(1, 1) == Catch::Approx(2.0f));
    CHECK(S(2, 2) == Catch::Approx(3.0f));
    CHECK(S(3, 3) == Catch::Approx(1.0f));
}
