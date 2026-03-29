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

TEST_CASE("AffineTransform * Vector (homogeneous)", "[transform][affine]") {
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

TEST_CASE("AffineTransform * Vector (affine action)", "[transform][affine]") {
    AffineTransform<float> xform;
    xform(0, 3) = 5.0f;
    xform(1, 3) = 10.0f;
    xform(2, 3) = 15.0f;

    // D-dimensional vector: affine action applies L*v + t
    Vector<float, 3> point({1.0f, 0.0f, 0.0f});
    Vector<float, 3> result = xform * point;

    CHECK(result(0) == Catch::Approx(6.0f));
    CHECK(result(1) == Catch::Approx(10.0f));
    CHECK(result(2) == Catch::Approx(15.0f));
}

TEST_CASE("AffineTransform affine action with rotation", "[transform][affine]") {
    // 90-degree rotation about Z + translation
    AffineTransform<float> xform;
    xform(0, 0) = 0.0f;  xform(0, 1) = -1.0f;
    xform(1, 0) = 1.0f;  xform(1, 1) = 0.0f;
    xform(0, 3) = 1.0f;
    xform(1, 3) = 2.0f;
    xform(2, 3) = 3.0f;

    Vector<float, 3> point({1.0f, 0.0f, 0.0f});
    Vector<float, 3> result = xform * point;

    // R * (1,0,0) = (0,1,0), then + (1,2,3)
    CHECK(result(0) == Catch::Approx(1.0f));
    CHECK(result(1) == Catch::Approx(3.0f));
    CHECK(result(2) == Catch::Approx(3.0f));
}

TEST_CASE("AffineTransform affine_inverse", "[transform][affine]") {
    // Build a translation
    AffineTransform<float> xform;
    xform(0, 3) = 3.0f;
    xform(1, 3) = 4.0f;
    xform(2, 3) = 5.0f;

    auto inv = xform.inverse();

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

TEST_CASE("Isometry inverse (rotation + translation)", "[transform][affine]") {
    // Build a 90-degree rotation about Z + translation as an Isometry
    Isometry<float> xform;
    xform(0, 0) = 0.0f;  xform(0, 1) = -1.0f;  // R = [0 -1; 1 0; 0 0 1]
    xform(1, 0) = 1.0f;  xform(1, 1) = 0.0f;
    xform(0, 3) = 3.0f;
    xform(1, 3) = 4.0f;
    xform(2, 3) = 5.0f;

    auto inv = xform.inverse();

    // R^T * R should be I, and translation should be -R^T * t
    auto identity = xform * inv;
    for (index_type r = 0; r < 4; ++r) {
        for (index_type c = 0; c < 4; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(identity(r, c) == Catch::Approx(expected).margin(1e-5f));
        }
    }

    // Also verify inv * xform = I
    auto identity2 = inv * xform;
    for (index_type r = 0; r < 4; ++r) {
        for (index_type c = 0; c < 4; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(identity2(r, c) == Catch::Approx(expected).margin(1e-5f));
        }
    }
}

TEST_CASE("Isometry inverse matches AffineTransform inverse for rigid body", "[transform][affine]") {
    // For a pure rotation + translation, Isometry and Affine inverses should agree
    AffineTransform<float> affine_xform;
    // 90-degree rotation about Y
    affine_xform(0, 0) = 0.0f;   affine_xform(0, 2) = 1.0f;
    affine_xform(2, 0) = -1.0f;  affine_xform(2, 2) = 0.0f;
    affine_xform(0, 3) = 7.0f;
    affine_xform(1, 3) = -3.0f;
    affine_xform(2, 3) = 11.0f;

    // Same transform as Isometry
    Isometry<float> iso_xform;
    iso_xform(0, 0) = 0.0f;   iso_xform(0, 2) = 1.0f;
    iso_xform(2, 0) = -1.0f;  iso_xform(2, 2) = 0.0f;
    iso_xform(0, 3) = 7.0f;
    iso_xform(1, 3) = -3.0f;
    iso_xform(2, 3) = 11.0f;

    auto inv_affine = affine_xform.inverse();
    auto inv_iso = iso_xform.inverse();

    for (index_type r = 0; r < 4; ++r) {
        for (index_type c = 0; c < 4; ++c) {
            CHECK(inv_iso(r, c) == Catch::Approx(inv_affine(r, c)).margin(1e-5f));
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

// ---- 2D AffineTransform tests ----

TEST_CASE("AffineTransform2D default is identity", "[transform][affine][2d]") {
    AffineTransform<float, 2> xform;

    for (index_type r = 0; r < 3; ++r) {
        for (index_type c = 0; c < 3; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(xform(r, c) == Catch::Approx(expected));
        }
    }
}

TEST_CASE("AffineTransform2D from matrix", "[transform][affine][2d]") {
    Matrix<float, 3, 3> m = expression::nullary::Identity<float, 3, 3>{};
    m(0, 2) = 1.0f;
    m(1, 2) = 2.0f;

    AffineTransform<float, 2> xform(m);

    CHECK(xform(0, 2) == Catch::Approx(1.0f));
    CHECK(xform(1, 2) == Catch::Approx(2.0f));
    CHECK(xform(2, 2) == Catch::Approx(1.0f));
}

TEST_CASE("AffineTransform2D linear view", "[transform][affine][2d]") {
    AffineTransform<float, 2> xform;
    // 90-degree rotation in 2D: [0 -1; 1 0]
    xform(0, 0) = 0.0f; xform(0, 1) = -1.0f;
    xform(1, 0) = 1.0f; xform(1, 1) = 0.0f;

    auto lin = xform.linear();

    CHECK(lin(0, 0) == Catch::Approx(0.0f));
    CHECK(lin(0, 1) == Catch::Approx(-1.0f));
    CHECK(lin(1, 0) == Catch::Approx(1.0f));
    CHECK(lin(1, 1) == Catch::Approx(0.0f));
}

TEST_CASE("AffineTransform2D translation view", "[transform][affine][2d]") {
    AffineTransform<float, 2> xform;
    xform(0, 2) = 10.0f;
    xform(1, 2) = 20.0f;

    auto t = xform.translation();

    CHECK(t(0) == Catch::Approx(10.0f));
    CHECK(t(1) == Catch::Approx(20.0f));
}

TEST_CASE("AffineTransform2D composition", "[transform][affine][2d]") {
    // T1: translate by (1, 0)
    AffineTransform<float, 2> t1;
    t1(0, 2) = 1.0f;

    // T2: translate by (0, 2)
    AffineTransform<float, 2> t2;
    t2(1, 2) = 2.0f;

    auto composed = t1 * t2;

    CHECK(composed(0, 2) == Catch::Approx(1.0f));
    CHECK(composed(1, 2) == Catch::Approx(2.0f));
    CHECK(composed(2, 2) == Catch::Approx(1.0f));
}

TEST_CASE("AffineTransform2D * Vector (homogeneous)", "[transform][affine][2d]") {
    AffineTransform<float, 2> xform;
    xform(0, 2) = 5.0f;
    xform(1, 2) = 10.0f;

    Vector<float, 3> point({1.0f, 0.0f, 1.0f});
    Vector<float, 3> result = Vector<float, 3>(xform * point);

    CHECK(result(0) == Catch::Approx(6.0f));
    CHECK(result(1) == Catch::Approx(10.0f));
    CHECK(result(2) == Catch::Approx(1.0f));
}

TEST_CASE("AffineTransform2D * Vector (affine action)", "[transform][affine][2d]") {
    AffineTransform<float, 2> xform;
    xform(0, 2) = 5.0f;
    xform(1, 2) = 10.0f;

    // D-dimensional vector: affine action applies L*v + t
    Vector<float, 2> point({1.0f, 0.0f});
    Vector<float, 2> result = xform * point;

    CHECK(result(0) == Catch::Approx(6.0f));
    CHECK(result(1) == Catch::Approx(10.0f));
}

TEST_CASE("AffineTransform2D affine action with rotation", "[transform][affine][2d]") {
    // 90-degree 2D rotation + translation
    AffineTransform<float, 2> xform;
    xform(0, 0) = 0.0f;  xform(0, 1) = -1.0f;
    xform(1, 0) = 1.0f;  xform(1, 1) = 0.0f;
    xform(0, 2) = 1.0f;
    xform(1, 2) = 2.0f;

    Vector<float, 2> point({1.0f, 0.0f});
    Vector<float, 2> result = xform * point;

    // R * (1,0) = (0,1), then + (1,2)
    CHECK(result(0) == Catch::Approx(1.0f));
    CHECK(result(1) == Catch::Approx(3.0f));
}

TEST_CASE("AffineTransform2D inverse", "[transform][affine][2d]") {
    AffineTransform<float, 2> xform;
    xform(0, 2) = 3.0f;
    xform(1, 2) = 4.0f;

    auto inv = xform.inverse();

    CHECK(inv(0, 2) == Catch::Approx(-3.0f));
    CHECK(inv(1, 2) == Catch::Approx(-4.0f));

    // T * T^-1 should be identity
    auto identity = xform * inv;
    for (index_type r = 0; r < 3; ++r) {
        for (index_type c = 0; c < 3; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(identity(r, c) == Catch::Approx(expected).margin(1e-5f));
        }
    }
}

TEST_CASE("Isometry2D inverse (rotation + translation)", "[transform][affine][2d]") {
    // 90-degree rotation + translation in 2D
    Isometry<float, 2> xform;
    xform(0, 0) = 0.0f;  xform(0, 1) = -1.0f;
    xform(1, 0) = 1.0f;  xform(1, 1) = 0.0f;
    xform(0, 2) = 3.0f;
    xform(1, 2) = 4.0f;

    auto inv = xform.inverse();

    auto identity = xform * inv;
    for (index_type r = 0; r < 3; ++r) {
        for (index_type c = 0; c < 3; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(identity(r, c) == Catch::Approx(expected).margin(1e-5f));
        }
    }

    auto identity2 = inv * xform;
    for (index_type r = 0; r < 3; ++r) {
        for (index_type c = 0; c < 3; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(identity2(r, c) == Catch::Approx(expected).margin(1e-5f));
        }
    }
}

TEST_CASE("AffineTransform2D with scale inverse", "[transform][affine][2d]") {
    // Non-uniform scale + translation in 2D
    AffineTransform<float, 2> xform;
    xform(0, 0) = 2.0f;
    xform(1, 1) = 3.0f;
    xform(0, 2) = 5.0f;
    xform(1, 2) = 7.0f;

    auto inv = xform.inverse();
    auto identity = xform * inv;

    for (index_type r = 0; r < 3; ++r) {
        for (index_type c = 0; c < 3; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(identity(r, c) == Catch::Approx(expected).margin(1e-5f));
        }
    }
}

TEST_CASE("AffineTransform2D to_matrix", "[transform][affine][2d]") {
    AffineTransform<float, 2> xform;
    xform(0, 2) = 1.0f;
    xform(1, 2) = 2.0f;

    Matrix<float, 3, 3> m = xform.to_matrix();

    CHECK(m(0, 0) == Catch::Approx(1.0f));
    CHECK(m(0, 2) == Catch::Approx(1.0f));
    CHECK(m(1, 2) == Catch::Approx(2.0f));
    CHECK(m(2, 2) == Catch::Approx(1.0f));
}

// ---- 1D AffineTransform tests ----

TEST_CASE("AffineTransform1D default is identity", "[transform][affine][1d]") {
    AffineTransform<float, 1> xform;

    CHECK(xform(0, 0) == Catch::Approx(1.0f));
    CHECK(xform(0, 1) == Catch::Approx(0.0f));
    CHECK(xform(1, 0) == Catch::Approx(0.0f));
    CHECK(xform(1, 1) == Catch::Approx(1.0f));
}

TEST_CASE("AffineTransform1D translation and inverse", "[transform][affine][1d]") {
    AffineTransform<float, 1> xform;
    xform(0, 1) = 5.0f;  // translate by 5

    auto inv = xform.inverse();
    CHECK(inv(0, 1) == Catch::Approx(-5.0f));

    auto identity = xform * inv;
    for (index_type r = 0; r < 2; ++r) {
        for (index_type c = 0; c < 2; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(identity(r, c) == Catch::Approx(expected).margin(1e-5f));
        }
    }
}

TEST_CASE("AffineTransform1D scale and inverse", "[transform][affine][1d]") {
    AffineTransform<float, 1> xform;
    xform(0, 0) = 3.0f;   // scale by 3
    xform(0, 1) = 2.0f;   // translate by 2

    auto inv = xform.inverse();
    auto identity = xform * inv;

    for (index_type r = 0; r < 2; ++r) {
        for (index_type c = 0; c < 2; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(identity(r, c) == Catch::Approx(expected).margin(1e-5f));
        }
    }
}
