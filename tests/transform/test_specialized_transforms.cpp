#include <cmath>
#include <numbers>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/transform/Translation.hpp>
#include <zipper/transform/Scaling.hpp>
#include <zipper/transform/Rotation.hpp>
#include <zipper/transform/AxisAngleRotation.hpp>
#include <zipper/transform/transform_compose.hpp>
#include <zipper/transform/quaternion_transform.hpp>

#include "../catch_include.hpp"

using namespace zipper;
using namespace zipper::transform;

// ============================================================================
// Concept checks (compile-time)
// ============================================================================

TEST_CASE("Specialized types satisfy Transform concept", "[transform][specialized][concepts]") {
    STATIC_CHECK(transform::concepts::Transform<Translation<float, 3>>);
    STATIC_CHECK(transform::concepts::Transform<Scaling<float, 3>>);
    STATIC_CHECK(transform::concepts::Transform<Rotation<float, 3>>);
    STATIC_CHECK(transform::concepts::Transform<AxisAngleRotation<float>>);

    // Specialized types are NOT MatrixTransforms
    STATIC_CHECK_FALSE(transform::concepts::MatrixTransform<Translation<float, 3>>);
    STATIC_CHECK_FALSE(transform::concepts::MatrixTransform<Scaling<float, 3>>);
    STATIC_CHECK_FALSE(transform::concepts::MatrixTransform<Rotation<float, 3>>);
    STATIC_CHECK_FALSE(transform::concepts::MatrixTransform<AxisAngleRotation<float>>);
}

TEST_CASE("Specialized types satisfy AffineTransform/Isometry concepts", "[transform][specialized][concepts]") {
    // Isometry types
    STATIC_CHECK(transform::concepts::AffineTransform<Translation<float, 3>>);
    STATIC_CHECK(transform::concepts::Isometry<Translation<float, 3>>);
    STATIC_CHECK(transform::concepts::AffineTransform<Rotation<float, 3>>);
    STATIC_CHECK(transform::concepts::Isometry<Rotation<float, 3>>);
    STATIC_CHECK(transform::concepts::AffineTransform<AxisAngleRotation<float>>);
    STATIC_CHECK(transform::concepts::Isometry<AxisAngleRotation<float>>);

    // Affine but not Isometry
    STATIC_CHECK(transform::concepts::AffineTransform<Scaling<float, 3>>);
    STATIC_CHECK_FALSE(transform::concepts::Isometry<Scaling<float, 3>>);
}

TEST_CASE("Specialized types expose dim", "[transform][specialized][concepts]") {
    STATIC_CHECK(Translation<float, 3>::dim == 3);
    STATIC_CHECK(Translation<float, 2>::dim == 2);
    STATIC_CHECK(Scaling<float, 3>::dim == 3);
    STATIC_CHECK(Rotation<float, 3>::dim == 3);
    STATIC_CHECK(AxisAngleRotation<float>::dim == 3);
}

// ============================================================================
// Translation
// ============================================================================

TEST_CASE("Translation default is identity", "[transform][specialized][translation]") {
    Translation<float, 3> t;
    CHECK(t(0) == 0.0f);
    CHECK(t(1) == 0.0f);
    CHECK(t(2) == 0.0f);
}

TEST_CASE("Translation from initializer list", "[transform][specialized][translation]") {
    Translation<float, 3> t({1.0f, 2.0f, 3.0f});
    CHECK(t(0) == 1.0f);
    CHECK(t(1) == 2.0f);
    CHECK(t(2) == 3.0f);
}

TEST_CASE("Translation .translation() and .linear()", "[transform][specialized][translation]") {
    Translation<float, 3> t({5.0f, 10.0f, 15.0f});

    // .translation() returns the internal vector
    auto tr = t.translation();
    CHECK(tr(0) == 5.0f);
    CHECK(tr(1) == 10.0f);
    CHECK(tr(2) == 15.0f);

    // .linear() returns identity
    auto lin = t.linear();
    for (index_type r = 0; r < 3; ++r) {
        for (index_type c = 0; c < 3; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(lin(r, c) == Catch::Approx(expected));
        }
    }
}

TEST_CASE("Translation inverse", "[transform][specialized][translation]") {
    Translation<float, 3> t({3.0f, -4.0f, 5.0f});
    auto inv = t.inverse();
    CHECK(inv(0) == Catch::Approx(-3.0f));
    CHECK(inv(1) == Catch::Approx(4.0f));
    CHECK(inv(2) == Catch::Approx(-5.0f));
}

TEST_CASE("Translation * Translation = sum", "[transform][specialized][translation]") {
    Translation<float, 3> a({1.0f, 2.0f, 3.0f});
    Translation<float, 3> b({4.0f, 5.0f, 6.0f});
    auto c = a * b;
    CHECK(c(0) == Catch::Approx(5.0f));
    CHECK(c(1) == Catch::Approx(7.0f));
    CHECK(c(2) == Catch::Approx(9.0f));
}

TEST_CASE("Translation * Vector = translated point", "[transform][specialized][translation]") {
    Translation<float, 3> t({1.0f, 2.0f, 3.0f});
    Vector<float, 3> p({10.0f, 20.0f, 30.0f});
    auto q = t * p;
    CHECK(q(0) == Catch::Approx(11.0f));
    CHECK(q(1) == Catch::Approx(22.0f));
    CHECK(q(2) == Catch::Approx(33.0f));
}

TEST_CASE("Translation to_transform and to_matrix", "[transform][specialized][translation]") {
    Translation<float, 3> t({1.0f, 2.0f, 3.0f});
    auto xform = t.to_transform();
    CHECK(xform(0, 3) == Catch::Approx(1.0f));
    CHECK(xform(1, 3) == Catch::Approx(2.0f));
    CHECK(xform(2, 3) == Catch::Approx(3.0f));
    // linear part is identity
    for (index_type i = 0; i < 3; ++i) {
        CHECK(xform(i, i) == Catch::Approx(1.0f));
    }

    auto mat = t.to_matrix();
    CHECK(mat(0, 3) == Catch::Approx(1.0f));
    CHECK(mat(3, 3) == Catch::Approx(1.0f));
}

TEST_CASE("Translation 2D", "[transform][specialized][translation]") {
    Translation<float, 2> t({5.0f, 10.0f});
    Vector<float, 2> p({1.0f, 2.0f});
    auto q = t * p;
    CHECK(q(0) == Catch::Approx(6.0f));
    CHECK(q(1) == Catch::Approx(12.0f));

    auto inv = t.inverse();
    auto identity_t = t * inv;
    CHECK(identity_t(0) == Catch::Approx(0.0f));
    CHECK(identity_t(1) == Catch::Approx(0.0f));
}

// ============================================================================
// Scaling
// ============================================================================

TEST_CASE("Scaling default is identity (uniform 1)", "[transform][specialized][scaling]") {
    Scaling<float, 3> s;
    CHECK(s(0) == 1.0f);
    CHECK(s(1) == 1.0f);
    CHECK(s(2) == 1.0f);
}

TEST_CASE("Scaling from initializer list", "[transform][specialized][scaling]") {
    Scaling<float, 3> s({2.0f, 3.0f, 4.0f});
    CHECK(s(0) == 2.0f);
    CHECK(s(1) == 3.0f);
    CHECK(s(2) == 4.0f);
}

TEST_CASE("Scaling uniform constructor", "[transform][specialized][scaling]") {
    Scaling<float, 3> s(5.0f);
    CHECK(s(0) == 5.0f);
    CHECK(s(1) == 5.0f);
    CHECK(s(2) == 5.0f);
}

TEST_CASE("Scaling .translation() and .linear()", "[transform][specialized][scaling]") {
    Scaling<float, 3> s({2.0f, 3.0f, 4.0f});

    // .translation() returns zero vector
    auto tr = s.translation();
    CHECK(tr(0) == 0.0f);
    CHECK(tr(1) == 0.0f);
    CHECK(tr(2) == 0.0f);

    // .linear() returns diagonal matrix with scale factors
    auto lin = s.linear();
    CHECK(lin(0, 0) == Catch::Approx(2.0f));
    CHECK(lin(1, 1) == Catch::Approx(3.0f));
    CHECK(lin(2, 2) == Catch::Approx(4.0f));
    CHECK(lin(0, 1) == Catch::Approx(0.0f));
    CHECK(lin(1, 0) == Catch::Approx(0.0f));
    CHECK(lin(0, 2) == Catch::Approx(0.0f));
}

TEST_CASE("Scaling inverse", "[transform][specialized][scaling]") {
    Scaling<float, 3> s({2.0f, 4.0f, 5.0f});
    auto inv = s.inverse();
    CHECK(inv(0) == Catch::Approx(0.5f));
    CHECK(inv(1) == Catch::Approx(0.25f));
    CHECK(inv(2) == Catch::Approx(0.2f));
}

TEST_CASE("Scaling * Scaling = component product", "[transform][specialized][scaling]") {
    Scaling<float, 3> a({2.0f, 3.0f, 4.0f});
    Scaling<float, 3> b({5.0f, 6.0f, 7.0f});
    auto c = a * b;
    CHECK(c(0) == Catch::Approx(10.0f));
    CHECK(c(1) == Catch::Approx(18.0f));
    CHECK(c(2) == Catch::Approx(28.0f));
}

TEST_CASE("Scaling * Vector = scaled point", "[transform][specialized][scaling]") {
    Scaling<float, 3> s({2.0f, 3.0f, 4.0f});
    Vector<float, 3> p({1.0f, 2.0f, 3.0f});
    auto q = s * p;
    CHECK(q(0) == Catch::Approx(2.0f));
    CHECK(q(1) == Catch::Approx(6.0f));
    CHECK(q(2) == Catch::Approx(12.0f));
}

TEST_CASE("Scaling to_transform", "[transform][specialized][scaling]") {
    Scaling<float, 3> s({2.0f, 3.0f, 4.0f});
    auto xform = s.to_transform();
    CHECK(xform(0, 0) == Catch::Approx(2.0f));
    CHECK(xform(1, 1) == Catch::Approx(3.0f));
    CHECK(xform(2, 2) == Catch::Approx(4.0f));
    CHECK(xform(0, 3) == Catch::Approx(0.0f));  // no translation
    CHECK(xform(3, 3) == Catch::Approx(1.0f));
}

// ============================================================================
// Rotation
// ============================================================================

TEST_CASE("Rotation default is identity", "[transform][specialized][rotation]") {
    Rotation<float, 3> r;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            CHECK(r(i, j) == Catch::Approx(expected));
        }
    }
}

TEST_CASE("Rotation .translation() and .linear()", "[transform][specialized][rotation]") {
    // 90-degree rotation about Z
    Rotation<float, 3> r;
    r(0, 0) = 0.0f;  r(0, 1) = -1.0f;
    r(1, 0) = 1.0f;  r(1, 1) = 0.0f;

    // .translation() returns zero
    auto tr = r.translation();
    CHECK(tr(0) == 0.0f);
    CHECK(tr(1) == 0.0f);
    CHECK(tr(2) == 0.0f);

    // .linear() returns the rotation matrix
    auto lin = r.linear();
    CHECK(lin(0, 0) == Catch::Approx(0.0f));
    CHECK(lin(0, 1) == Catch::Approx(-1.0f));
    CHECK(lin(1, 0) == Catch::Approx(1.0f));
    CHECK(lin(1, 1) == Catch::Approx(0.0f));
    CHECK(lin(2, 2) == Catch::Approx(1.0f));
}

TEST_CASE("Rotation inverse is transpose", "[transform][specialized][rotation]") {
    // 90-degree rotation about Z
    Rotation<float, 3> r;
    r(0, 0) = 0.0f;  r(0, 1) = -1.0f;
    r(1, 0) = 1.0f;  r(1, 1) = 0.0f;

    auto inv = r.inverse();
    // R^T: transposed
    CHECK(inv(0, 0) == Catch::Approx(0.0f));
    CHECK(inv(0, 1) == Catch::Approx(1.0f));
    CHECK(inv(1, 0) == Catch::Approx(-1.0f));
    CHECK(inv(1, 1) == Catch::Approx(0.0f));

    // R * R^T = I
    auto identity = r * inv;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            CHECK(identity(i, j) == Catch::Approx(expected).margin(1e-5f));
        }
    }
}

TEST_CASE("Rotation * Vector = rotated point", "[transform][specialized][rotation]") {
    // 90-degree rotation about Z
    Rotation<float, 3> r;
    r(0, 0) = 0.0f;  r(0, 1) = -1.0f;
    r(1, 0) = 1.0f;  r(1, 1) = 0.0f;

    Vector<float, 3> p({1.0f, 0.0f, 0.0f});
    auto q = r * p;
    CHECK(q(0) == Catch::Approx(0.0f));
    CHECK(q(1) == Catch::Approx(1.0f));
    CHECK(q(2) == Catch::Approx(0.0f));
}

TEST_CASE("Rotation * Rotation = composed rotation", "[transform][specialized][rotation]") {
    // Two 90-degree rotations about Z → 180-degree rotation
    Rotation<float, 3> r;
    r(0, 0) = 0.0f;  r(0, 1) = -1.0f;
    r(1, 0) = 1.0f;  r(1, 1) = 0.0f;

    auto r180 = r * r;
    // 180-degree about Z: [-1 0; 0 -1; 0 0 1]
    CHECK(r180(0, 0) == Catch::Approx(-1.0f).margin(1e-5f));
    CHECK(r180(1, 1) == Catch::Approx(-1.0f).margin(1e-5f));
    CHECK(r180(2, 2) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(r180(0, 1) == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Rotation to_transform", "[transform][specialized][rotation]") {
    Rotation<float, 3> r;
    r(0, 0) = 0.0f;  r(0, 1) = -1.0f;
    r(1, 0) = 1.0f;  r(1, 1) = 0.0f;

    auto xform = r.to_transform();
    CHECK(xform(0, 0) == Catch::Approx(0.0f));
    CHECK(xform(0, 1) == Catch::Approx(-1.0f));
    CHECK(xform(1, 0) == Catch::Approx(1.0f));
    CHECK(xform(0, 3) == Catch::Approx(0.0f));  // no translation
    CHECK(xform(3, 3) == Catch::Approx(1.0f));
}

// ============================================================================
// AxisAngleRotation
// ============================================================================

TEST_CASE("AxisAngleRotation default is identity", "[transform][specialized][axisangle]") {
    AxisAngleRotation<float> aa;
    CHECK(aa.angle() == 0.0f);

    // Should act as identity on a vector
    Vector<float, 3> p({1.0f, 2.0f, 3.0f});
    auto q = aa * p;
    CHECK(q(0) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(q(1) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(q(2) == Catch::Approx(3.0f).margin(1e-5f));
}

TEST_CASE("AxisAngleRotation 90 degrees about Z", "[transform][specialized][axisangle]") {
    float pi = std::numbers::pi_v<float>;
    AxisAngleRotation<float> aa(pi / 2.0f, {0.0f, 0.0f, 1.0f});

    Vector<float, 3> p({1.0f, 0.0f, 0.0f});
    auto q = aa * p;
    CHECK(q(0) == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(q(1) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(q(2) == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("AxisAngleRotation .translation() and .linear()", "[transform][specialized][axisangle]") {
    float pi = std::numbers::pi_v<float>;
    AxisAngleRotation<float> aa(pi / 2.0f, {0.0f, 0.0f, 1.0f});

    // .translation() returns zero
    auto tr = aa.translation();
    CHECK(tr(0) == 0.0f);
    CHECK(tr(1) == 0.0f);
    CHECK(tr(2) == 0.0f);

    // .linear() returns the 3x3 rotation matrix
    auto lin = aa.linear();
    CHECK(lin(0, 0) == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(lin(0, 1) == Catch::Approx(-1.0f).margin(1e-5f));
    CHECK(lin(1, 0) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(lin(1, 1) == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(lin(2, 2) == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("AxisAngleRotation inverse", "[transform][specialized][axisangle]") {
    float pi = std::numbers::pi_v<float>;
    AxisAngleRotation<float> aa(pi / 4.0f, {0.0f, 1.0f, 0.0f});

    auto inv = aa.inverse();
    CHECK(inv.angle() == Catch::Approx(-pi / 4.0f));

    // aa * inv should be identity action
    Vector<float, 3> p({1.0f, 2.0f, 3.0f});
    auto q = aa * p;
    auto back = inv * q;
    CHECK(back(0) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(back(1) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(back(2) == Catch::Approx(3.0f).margin(1e-5f));
}

TEST_CASE("AxisAngleRotation to_rotation", "[transform][specialized][axisangle]") {
    float pi = std::numbers::pi_v<float>;
    AxisAngleRotation<float> aa(pi / 2.0f, {0.0f, 0.0f, 1.0f});

    auto rot = aa.to_rotation();
    CHECK(rot(0, 0) == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(rot(0, 1) == Catch::Approx(-1.0f).margin(1e-5f));
    CHECK(rot(1, 0) == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("AxisAngleRotation * Rotation cross-type", "[transform][specialized][axisangle]") {
    float pi = std::numbers::pi_v<float>;
    AxisAngleRotation<float> aa(pi / 2.0f, {0.0f, 0.0f, 1.0f});

    Rotation<float, 3> r;
    r(0, 0) = 0.0f;  r(0, 1) = -1.0f;
    r(1, 0) = 1.0f;  r(1, 1) = 0.0f;

    // Both are 90-degree Z rotations → result should be 180-degree
    auto composed = aa * r;
    CHECK(composed(0, 0) == Catch::Approx(-1.0f).margin(1e-5f));
    CHECK(composed(1, 1) == Catch::Approx(-1.0f).margin(1e-5f));
    CHECK(composed(2, 2) == Catch::Approx(1.0f).margin(1e-5f));
}

// ============================================================================
// Cross-type composition
// ============================================================================

TEST_CASE("Translation * Rotation → promoted Transform", "[transform][specialized][compose]") {
    Translation<float, 3> t({1.0f, 2.0f, 3.0f});

    Rotation<float, 3> r;
    r(0, 0) = 0.0f;  r(0, 1) = -1.0f;
    r(1, 0) = 1.0f;  r(1, 1) = 0.0f;

    auto composed = t * r;

    // Should be [R | t; 0 1] since T * R: rotate then translate
    // Wait — T * R means apply R first then T:
    // (T * R) * v = T * (R * v) = R*v + t

    Vector<float, 3> p({1.0f, 0.0f, 0.0f});
    // R * (1,0,0) = (0,1,0), then + (1,2,3) = (1,3,3)
    auto q = composed * p;
    CHECK(q(0) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(q(1) == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(q(2) == Catch::Approx(3.0f).margin(1e-5f));
}

TEST_CASE("Rotation * Translation → promoted Transform", "[transform][specialized][compose]") {
    Rotation<float, 3> r;
    r(0, 0) = 0.0f;  r(0, 1) = -1.0f;
    r(1, 0) = 1.0f;  r(1, 1) = 0.0f;

    Translation<float, 3> t({1.0f, 2.0f, 3.0f});

    auto composed = r * t;

    // R * T: translate then rotate
    // (R * T) * v = R * (v + t) = R*v + R*t
    Vector<float, 3> p({0.0f, 0.0f, 0.0f});
    // R * t = R * (1,2,3) = (-2, 1, 3)
    auto q = composed * p;
    CHECK(q(0) == Catch::Approx(-2.0f).margin(1e-5f));
    CHECK(q(1) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(q(2) == Catch::Approx(3.0f).margin(1e-5f));
}

TEST_CASE("Scaling * Translation → promoted Transform", "[transform][specialized][compose]") {
    Scaling<float, 3> s({2.0f, 3.0f, 4.0f});
    Translation<float, 3> t({1.0f, 1.0f, 1.0f});

    auto composed = s * t;

    // S * T: translate then scale
    // (S * T) * v = S * (v + t) = S*v + S*t
    Vector<float, 3> p({0.0f, 0.0f, 0.0f});
    auto q = composed * p;
    CHECK(q(0) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(q(1) == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(q(2) == Catch::Approx(4.0f).margin(1e-5f));
}

TEST_CASE("Specialized * MatrixTransform composition", "[transform][specialized][compose]") {
    Translation<float, 3> t({1.0f, 2.0f, 3.0f});

    AffineTransform<float> xform;
    xform(0, 0) = 2.0f;  // scale x by 2

    auto composed = t * xform;

    // t * xform: xform first, then translate
    Vector<float, 3> p({1.0f, 0.0f, 0.0f});
    auto q = composed * p;
    // xform: (2, 0, 0), then + (1, 2, 3) = (3, 2, 3)
    CHECK(q(0) == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(q(1) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(q(2) == Catch::Approx(3.0f).margin(1e-5f));
}

TEST_CASE("MatrixTransform * Specialized composition", "[transform][specialized][compose]") {
    AffineTransform<float> xform;
    xform(0, 0) = 2.0f;  // scale x by 2

    Translation<float, 3> t({1.0f, 2.0f, 3.0f});

    auto composed = xform * t;

    // xform * t: translate first, then xform
    Vector<float, 3> p({0.0f, 0.0f, 0.0f});
    auto q = composed * p;
    // t: (1, 2, 3), then xform: (2*1, 2, 3) = (2, 2, 3)
    CHECK(q(0) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(q(1) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(q(2) == Catch::Approx(3.0f).margin(1e-5f));
}

// ============================================================================
// to_quaternion for specialized Rotation type
// ============================================================================

TEST_CASE("to_quaternion works with Rotation type", "[transform][specialized][quaternion]") {
    // 90-degree rotation about Z
    Rotation<float, 3> r;
    r(0, 0) = 0.0f;  r(0, 1) = -1.0f;
    r(1, 0) = 1.0f;  r(1, 1) = 0.0f;

    auto q = to_quaternion(r);

    // For 90-degree Z rotation: q = (cos(45°), 0, 0, sin(45°))
    float expected_w = std::cos(std::numbers::pi_v<float> / 4.0f);
    float expected_z = std::sin(std::numbers::pi_v<float> / 4.0f);
    CHECK(q.w() == Catch::Approx(expected_w).margin(1e-5f));
    CHECK(q.x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(q.y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(q.z() == Catch::Approx(expected_z).margin(1e-5f));
}

TEST_CASE("to_quaternion works with AxisAngleRotation type", "[transform][specialized][quaternion]") {
    float pi = std::numbers::pi_v<float>;
    AxisAngleRotation<float> aa(pi / 2.0f, {0.0f, 1.0f, 0.0f});

    auto q = to_quaternion(aa);

    // For 90-degree Y rotation: q = (cos(45°), 0, sin(45°), 0)
    float expected_w = std::cos(pi / 4.0f);
    float expected_y = std::sin(pi / 4.0f);
    CHECK(q.w() == Catch::Approx(expected_w).margin(1e-5f));
    CHECK(q.x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(q.y() == Catch::Approx(expected_y).margin(1e-5f));
    CHECK(q.z() == Catch::Approx(0.0f).margin(1e-5f));
}
