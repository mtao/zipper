#include <cmath>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/transform/AffineTransform.hpp>

#include "../catch_include.hpp"

using namespace zipper;
using namespace zipper::transform;

TEST_CASE("AffineTransform default is identity", "[transform][affine]") {
    AffineTransform<float> xform;

    for (index_type r = 0; r < 4; ++r) {
        for (index_type c = 0; c < 4; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(xform(r, c) == Catch::Approx(expected));
        }
    }
}

TEST_CASE("AffineTransform from matrix", "[transform][affine]") {
    Matrix<float, 4, 4> m = expression::nullary::Identity<float, 4, 4>{};
    m(0, 3) = 1.0f;
    m(1, 3) = 2.0f;
    m(2, 3) = 3.0f;

    AffineTransform<float> xform(m);

    CHECK(xform(0, 3) == Catch::Approx(1.0f));
    CHECK(xform(1, 3) == Catch::Approx(2.0f));
    CHECK(xform(2, 3) == Catch::Approx(3.0f));
    CHECK(xform(3, 3) == Catch::Approx(1.0f));
}

TEST_CASE("AffineTransform linear view", "[transform][affine]") {
    AffineTransform<float> xform;
    // Set a rotation-like 3x3 block
    xform(0, 0) = 0.0f; xform(0, 1) = -1.0f;
    xform(1, 0) = 1.0f; xform(1, 1) = 0.0f;

    auto lin = xform.linear();

    CHECK(lin(0, 0) == Catch::Approx(0.0f));
    CHECK(lin(0, 1) == Catch::Approx(-1.0f));
    CHECK(lin(1, 0) == Catch::Approx(1.0f));
    CHECK(lin(1, 1) == Catch::Approx(0.0f));
    CHECK(lin(2, 2) == Catch::Approx(1.0f));
}

TEST_CASE("AffineTransform translation view", "[transform][affine]") {
    AffineTransform<float> xform;
    xform(0, 3) = 10.0f;
    xform(1, 3) = 20.0f;
    xform(2, 3) = 30.0f;

    auto t = xform.translation();

    CHECK(t(0) == Catch::Approx(10.0f));
    CHECK(t(1) == Catch::Approx(20.0f));
    CHECK(t(2) == Catch::Approx(30.0f));
}

TEST_CASE("AffineTransform composition", "[transform][affine]") {
    // T1: translate by (1, 0, 0)
    AffineTransform<float> t1;
    t1(0, 3) = 1.0f;

    // T2: translate by (0, 2, 0)
    AffineTransform<float> t2;
    t2(0, 3) = 0.0f;
    t2(1, 3) = 2.0f;

    auto composed = t1 * t2;

    // Combined translation should be (1, 2, 0)
    CHECK(composed(0, 3) == Catch::Approx(1.0f));
    CHECK(composed(1, 3) == Catch::Approx(2.0f));
    CHECK(composed(2, 3) == Catch::Approx(0.0f));
    CHECK(composed(3, 3) == Catch::Approx(1.0f));
}

TEST_CASE("AffineTransform * Vector", "[transform][affine]") {
    AffineTransform<float> xform;
    xform(0, 3) = 5.0f;
    xform(1, 3) = 10.0f;
    xform(2, 3) = 15.0f;

    Vector<float, 4> point({1.0f, 0.0f, 0.0f, 1.0f});
    Vector<float, 4> result = Vector<float, 4>(xform * point);

    CHECK(result(0) == Catch::Approx(6.0f));
    CHECK(result(1) == Catch::Approx(10.0f));
    CHECK(result(2) == Catch::Approx(15.0f));
    CHECK(result(3) == Catch::Approx(1.0f));
}

TEST_CASE("AffineTransform affine_inverse", "[transform][affine]") {
    // Build a translation
    AffineTransform<float> xform;
    xform(0, 3) = 3.0f;
    xform(1, 3) = 4.0f;
    xform(2, 3) = 5.0f;

    auto inv = xform.affine_inverse();

    // Inverse of translation (3,4,5) is translation (-3,-4,-5)
    CHECK(inv(0, 3) == Catch::Approx(-3.0f));
    CHECK(inv(1, 3) == Catch::Approx(-4.0f));
    CHECK(inv(2, 3) == Catch::Approx(-5.0f));

    // T * T^-1 should be identity
    auto identity = xform * inv;
    for (index_type r = 0; r < 4; ++r) {
        for (index_type c = 0; c < 4; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(identity(r, c) == Catch::Approx(expected).margin(1e-5f));
        }
    }
}

TEST_CASE("AffineTransform to_matrix", "[transform][affine]") {
    AffineTransform<float> xform;
    xform(0, 3) = 1.0f;
    xform(1, 3) = 2.0f;

    Matrix<float, 4, 4> m = xform.to_matrix();

    CHECK(m(0, 0) == Catch::Approx(1.0f));
    CHECK(m(0, 3) == Catch::Approx(1.0f));
    CHECK(m(1, 3) == Catch::Approx(2.0f));
    CHECK(m(3, 3) == Catch::Approx(1.0f));
}
