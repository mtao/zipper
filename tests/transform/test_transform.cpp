#include <cmath>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/transform/Transform.hpp>
#include <zipper/transform/model.hpp>
#include <zipper/transform/projection.hpp>
#include <zipper/transform/common.hpp>

#include "../catch_include.hpp"

using namespace zipper;
using namespace zipper::transform;

// ============================================================================
// TransformMode enum and promote_mode
// ============================================================================

TEST_CASE("promote_mode returns least restrictive", "[transform][mode]") {
    CHECK(promote_mode(TransformMode::Isometry, TransformMode::Isometry) == TransformMode::Isometry);
    CHECK(promote_mode(TransformMode::Isometry, TransformMode::Affine) == TransformMode::Affine);
    CHECK(promote_mode(TransformMode::Affine, TransformMode::Isometry) == TransformMode::Affine);
    CHECK(promote_mode(TransformMode::Affine, TransformMode::Affine) == TransformMode::Affine);
    CHECK(promote_mode(TransformMode::Isometry, TransformMode::Projective) == TransformMode::Projective);
    CHECK(promote_mode(TransformMode::Projective, TransformMode::Isometry) == TransformMode::Projective);
    CHECK(promote_mode(TransformMode::Affine, TransformMode::Projective) == TransformMode::Projective);
    CHECK(promote_mode(TransformMode::Projective, TransformMode::Projective) == TransformMode::Projective);
}

// ============================================================================
// Isometry type
// ============================================================================

TEST_CASE("Isometry default is identity", "[transform][isometry]") {
    Isometry<float> xform;
    for (index_type r = 0; r < 4; ++r) {
        for (index_type c = 0; c < 4; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(xform(r, c) == Catch::Approx(expected));
        }
    }
}

TEST_CASE("Isometry has correct mode", "[transform][isometry]") {
    Isometry<float> xform;
    STATIC_CHECK(std::decay_t<decltype(xform)>::mode == TransformMode::Isometry);
}

TEST_CASE("Isometry inverse uses rotation transpose", "[transform][isometry]") {
    // 90-degree rotation about Z + translation
    Isometry<float> xform;
    xform(0, 0) = 0.0f;  xform(0, 1) = -1.0f;
    xform(1, 0) = 1.0f;  xform(1, 1) = 0.0f;
    xform(0, 3) = 3.0f;
    xform(1, 3) = 4.0f;
    xform(2, 3) = 5.0f;

    auto inv = xform.inverse();  // dispatches to rotation_inverse

    auto identity = xform * inv;
    for (index_type r = 0; r < 4; ++r) {
        for (index_type c = 0; c < 4; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(identity(r, c) == Catch::Approx(expected).margin(1e-5f));
        }
    }
}

TEST_CASE("Isometry * Vector<D> is affine action", "[transform][isometry]") {
    Isometry<float> xform;
    xform(0, 3) = 5.0f;
    xform(1, 3) = 10.0f;
    xform(2, 3) = 15.0f;

    Vector<float, 3> point({1.0f, 0.0f, 0.0f});
    Vector<float, 3> result = xform * point;

    CHECK(result(0) == Catch::Approx(6.0f));
    CHECK(result(1) == Catch::Approx(10.0f));
    CHECK(result(2) == Catch::Approx(15.0f));
}

// ============================================================================
// AffineTransform (Affine mode)
// ============================================================================

TEST_CASE("AffineTransform has correct mode", "[transform][affine]") {
    AffineTransform<float> xform;
    STATIC_CHECK(std::decay_t<decltype(xform)>::mode == TransformMode::Affine);
}

TEST_CASE("AffineTransform inverse uses affine_inverse", "[transform][affine]") {
    // Non-uniform scale + translation
    AffineTransform<float> xform;
    xform(0, 0) = 2.0f;
    xform(1, 1) = 3.0f;
    xform(0, 3) = 5.0f;
    xform(1, 3) = 7.0f;

    auto inv = xform.inverse();  // dispatches to affine_inverse

    auto identity = xform * inv;
    for (index_type r = 0; r < 4; ++r) {
        for (index_type c = 0; c < 4; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(identity(r, c) == Catch::Approx(expected).margin(1e-5f));
        }
    }
}

// ============================================================================
// ProjectiveTransform
// ============================================================================

TEST_CASE("ProjectiveTransform has correct mode", "[transform][projective]") {
    ProjectiveTransform<float> xform;
    STATIC_CHECK(std::decay_t<decltype(xform)>::mode == TransformMode::Projective);
}

TEST_CASE("ProjectiveTransform inverse uses full matrix inverse", "[transform][projective]") {
    // Build a perspective-like projective matrix
    auto proj = perspective(radians(45.0f), 16.0f / 9.0f, 0.1f, 100.0f);

    auto inv = proj.inverse();  // dispatches to projective_inverse

    // proj * inv should be close to identity
    // Use to_matrix() for the multiplication since both are projective
    auto identity = proj * inv;
    for (index_type r = 0; r < 4; ++r) {
        for (index_type c = 0; c < 4; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(identity(r, c) == Catch::Approx(expected).margin(1e-4f));
        }
    }
}

TEST_CASE("ProjectiveTransform * Vector<D> does perspective division", "[transform][projective]") {
    // A simple projective transform: scale x,y and set w=z
    ProjectiveTransform<float, 3> xform;
    xform(0, 0) = 1.0f;
    xform(1, 1) = 1.0f;
    xform(2, 2) = 1.0f;
    xform(3, 2) = 1.0f;   // w = z
    xform(3, 3) = 0.0f;   // not identity last row

    // Apply to (2, 4, 2): homogeneous result is (2, 4, 2, 2)
    // After perspective division: (1, 2, 1)
    Vector<float, 3> point({2.0f, 4.0f, 2.0f});
    Vector<float, 3> result = xform * point;

    CHECK(result(0) == Catch::Approx(1.0f));
    CHECK(result(1) == Catch::Approx(2.0f));
    CHECK(result(2) == Catch::Approx(1.0f));
}

// ============================================================================
// Mode promotion on composition
// ============================================================================

TEST_CASE("Isometry * Isometry -> Isometry", "[transform][mode][composition]") {
    Isometry<float> a;
    a(0, 3) = 1.0f;
    Isometry<float> b;
    b(1, 3) = 2.0f;

    auto result = a * b;
    STATIC_CHECK(std::decay_t<decltype(result)>::mode == TransformMode::Isometry);

    CHECK(result(0, 3) == Catch::Approx(1.0f));
    CHECK(result(1, 3) == Catch::Approx(2.0f));
}

TEST_CASE("Isometry * AffineTransform -> AffineTransform", "[transform][mode][composition]") {
    Isometry<float> a;
    a(0, 3) = 1.0f;
    AffineTransform<float> b;
    b(0, 0) = 2.0f;  // scale

    auto result = a * b;
    STATIC_CHECK(std::decay_t<decltype(result)>::mode == TransformMode::Affine);
}

TEST_CASE("AffineTransform * Isometry -> AffineTransform", "[transform][mode][composition]") {
    AffineTransform<float> a;
    a(0, 0) = 2.0f;
    Isometry<float> b;
    b(0, 3) = 1.0f;

    auto result = a * b;
    STATIC_CHECK(std::decay_t<decltype(result)>::mode == TransformMode::Affine);
}

TEST_CASE("AffineTransform * ProjectiveTransform -> ProjectiveTransform", "[transform][mode][composition]") {
    AffineTransform<float> a;
    a(0, 3) = 1.0f;
    auto proj = perspective(radians(45.0f), 1.0f, 0.1f, 100.0f);

    auto result = a * proj;
    STATIC_CHECK(std::decay_t<decltype(result)>::mode == TransformMode::Projective);
}

TEST_CASE("Isometry * ProjectiveTransform -> ProjectiveTransform", "[transform][mode][composition]") {
    Isometry<float> a;
    a(0, 3) = 1.0f;
    auto proj = perspective(radians(45.0f), 1.0f, 0.1f, 100.0f);

    auto result = a * proj;
    STATIC_CHECK(std::decay_t<decltype(result)>::mode == TransformMode::Projective);
}

// ============================================================================
// Cross-mode construction and assignment
// ============================================================================

TEST_CASE("ProjectiveTransform from Isometry", "[transform][conversion]") {
    Isometry<float> iso;
    iso(0, 3) = 5.0f;

    ProjectiveTransform<float> proj(iso);
    CHECK(proj(0, 3) == Catch::Approx(5.0f));
    CHECK(proj(0, 0) == Catch::Approx(1.0f));
}

TEST_CASE("AffineTransform from Isometry", "[transform][conversion]") {
    Isometry<float> iso;
    iso(0, 3) = 5.0f;

    AffineTransform<float> aff(iso);
    CHECK(aff(0, 3) == Catch::Approx(5.0f));
}

// ============================================================================
// Model factory functions return correct modes
// ============================================================================

TEST_CASE("translation returns Isometry", "[transform][model][mode]") {
    auto T = translation(Vector<float, 3>({1.0f, 2.0f, 3.0f}));
    STATIC_CHECK(std::decay_t<decltype(T)>::mode == TransformMode::Isometry);
}

TEST_CASE("rotation returns Isometry", "[transform][model][mode]") {
    auto R = rotation(radians(45.0f), Vector<float, 3>({0.0f, 1.0f, 0.0f}));
    STATIC_CHECK(std::decay_t<decltype(R)>::mode == TransformMode::Isometry);
}

TEST_CASE("scaling returns Affine", "[transform][model][mode]") {
    auto S = scaling(Vector<float, 3>({2.0f, 2.0f, 2.0f}));
    STATIC_CHECK(std::decay_t<decltype(S)>::mode == TransformMode::Affine);
}

TEST_CASE("translation * rotation * scaling -> Affine", "[transform][model][mode]") {
    auto T = translation(Vector<float, 3>({1.0f, 0.0f, 0.0f}));
    auto R = rotation(radians(90.0f), Vector<float, 3>({0.0f, 0.0f, 1.0f}));
    auto S = scaling(Vector<float, 3>({2.0f, 2.0f, 2.0f}));

    auto model = T * R * S;
    STATIC_CHECK(std::decay_t<decltype(model)>::mode == TransformMode::Affine);

    // T * R is Isometry
    auto TR = T * R;
    STATIC_CHECK(std::decay_t<decltype(TR)>::mode == TransformMode::Isometry);
}

// ============================================================================
// 2D transforms with modes
// ============================================================================

TEST_CASE("Isometry 2D inverse", "[transform][isometry][2d]") {
    Isometry<float, 2> xform;
    // 90-degree rotation + translation
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
}

TEST_CASE("AffineTransform 2D inverse", "[transform][affine][2d]") {
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

// ============================================================================
// Projective composition uses full matmul (not block formula)
// ============================================================================

TEST_CASE("Projective composition is full matmul", "[transform][projective][composition]") {
    // Two projective transforms with non-trivial last rows
    ProjectiveTransform<float, 2> a;
    a(0, 0) = 1.0f; a(0, 1) = 0.0f; a(0, 2) = 0.0f;
    a(1, 0) = 0.0f; a(1, 1) = 1.0f; a(1, 2) = 0.0f;
    a(2, 0) = 1.0f; a(2, 1) = 0.0f; a(2, 2) = 1.0f;  // last row = [1 0 1]

    ProjectiveTransform<float, 2> b;
    b(0, 0) = 2.0f; b(0, 1) = 0.0f; b(0, 2) = 1.0f;
    b(1, 0) = 0.0f; b(1, 1) = 3.0f; b(1, 2) = 0.0f;
    b(2, 0) = 0.0f; b(2, 1) = 0.0f; b(2, 2) = 1.0f;

    auto result = a * b;

    // Manual: a * b (3x3 full matmul)
    // row 0: [1*2+0*0+0*0, 1*0+0*3+0*0, 1*1+0*0+0*1] = [2, 0, 1]
    // row 1: [0*2+1*0+0*0, 0*0+1*3+0*0, 0*1+1*0+0*1] = [0, 3, 0]
    // row 2: [1*2+0*0+1*0, 1*0+0*3+1*0, 1*1+0*0+1*1] = [2, 0, 2]
    CHECK(result(0, 0) == Catch::Approx(2.0f));
    CHECK(result(0, 2) == Catch::Approx(1.0f));
    CHECK(result(1, 1) == Catch::Approx(3.0f));
    CHECK(result(2, 0) == Catch::Approx(2.0f));
    CHECK(result(2, 2) == Catch::Approx(2.0f));
}

// ============================================================================
// Vector<D+1> homogeneous multiply works for all modes
// ============================================================================

TEST_CASE("Isometry * Vector<D+1> is homogeneous multiply", "[transform][isometry]") {
    Isometry<float> xform;
    xform(0, 3) = 5.0f;

    Vector<float, 4> v({1.0f, 0.0f, 0.0f, 1.0f});
    Vector<float, 4> result = Vector<float, 4>(xform * v);

    CHECK(result(0) == Catch::Approx(6.0f));
    CHECK(result(3) == Catch::Approx(1.0f));
}

TEST_CASE("ProjectiveTransform * Vector<D+1> is homogeneous multiply (no division)", "[transform][projective]") {
    ProjectiveTransform<float, 3> xform;
    xform(3, 2) = 1.0f;  // w = z
    xform(3, 3) = 0.0f;

    Vector<float, 4> v({1.0f, 2.0f, 3.0f, 1.0f});
    Vector<float, 4> result = Vector<float, 4>(xform * v);

    // w should be z=3, not divided
    CHECK(result(0) == Catch::Approx(1.0f));
    CHECK(result(3) == Catch::Approx(3.0f));
}

// ============================================================================
// concepts::Transform, concepts::AffineTransform, concepts::Isometry
// ============================================================================

TEST_CASE("concepts::Transform matches all modes", "[transform][concepts]") {
    STATIC_CHECK(transform::concepts::Transform<Isometry<float>>);
    STATIC_CHECK(transform::concepts::Transform<AffineTransform<float>>);
    STATIC_CHECK(transform::concepts::Transform<ProjectiveTransform<float>>);
    STATIC_CHECK(transform::concepts::Transform<Transform<float>>);
    STATIC_CHECK(transform::concepts::Transform<Isometry<float, 2>>);
}

TEST_CASE("concepts::AffineTransform matches Affine and Isometry only", "[transform][concepts]") {
    STATIC_CHECK(transform::concepts::AffineTransform<Isometry<float>>);
    STATIC_CHECK(transform::concepts::AffineTransform<AffineTransform<float>>);
    STATIC_CHECK_FALSE(transform::concepts::AffineTransform<ProjectiveTransform<float>>);
    STATIC_CHECK(transform::concepts::AffineTransform<Isometry<float, 2>>);
    STATIC_CHECK(transform::concepts::AffineTransform<AffineTransform<float, 2>>);
}

TEST_CASE("concepts::Isometry matches only Isometry mode", "[transform][concepts]") {
    STATIC_CHECK(transform::concepts::Isometry<Isometry<float>>);
    STATIC_CHECK_FALSE(transform::concepts::Isometry<AffineTransform<float>>);
    STATIC_CHECK_FALSE(transform::concepts::Isometry<ProjectiveTransform<float>>);
    STATIC_CHECK(transform::concepts::Isometry<Isometry<float, 2>>);
}
