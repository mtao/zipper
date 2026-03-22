#include <cmath>
#include <numbers>
#include <tuple>

#include <zipper/Matrix.hpp>
#include <zipper/Quaternion.hpp>
#include <zipper/Vector.hpp>
#include <zipper/transform/decompose.hpp>
#include <zipper/transform/model.hpp>
#include <zipper/transform/common.hpp>
#include <zipper/transform/quaternion_transform.hpp>
#include <zipper/transform/Transform.hpp>

#include "../catch_include.hpp"

using namespace zipper;
using namespace zipper::transform;

// ── Helpers ─────────────────────────────────────────────────────────

/// Check that a Translation matches expected values within tolerance.
template<typename T, index_type D>
static void check_translation(const Translation<T, D> &t,
                              std::initializer_list<T> expected,
                              T margin = T(1e-5)) {
    auto it = expected.begin();
    for (index_type i = 0; i < D && it != expected.end(); ++i, ++it) {
        CHECK(static_cast<T>(t(i)) == Catch::Approx(*it).margin(margin));
    }
}

/// Check that Scaling matches expected values within tolerance.
template<typename T, index_type D>
static void check_scaling(const Scaling<T, D> &s,
                          std::initializer_list<T> expected,
                          T margin = T(1e-5)) {
    auto it = expected.begin();
    for (index_type i = 0; i < D && it != expected.end(); ++i, ++it) {
        CHECK(static_cast<T>(s(i)) == Catch::Approx(*it).margin(margin));
    }
}

/// Check that a Rotation matches another rotation matrix within tolerance.
template<typename T, index_type D>
static void check_rotation(const Rotation<T, D> &r,
                           const Rotation<T, D> &expected,
                           T margin = T(1e-5)) {
    for (index_type row = 0; row < D; ++row) {
        for (index_type col = 0; col < D; ++col) {
            CHECK(static_cast<T>(r(row, col)) == Catch::Approx(static_cast<T>(expected(row, col))).margin(margin));
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

TEST_CASE("trs_decompose identity", "[transform][decompose]") {
    AffineTransform<float> xf;// identity
    auto [t, r, s] = trs_decompose(xf);

    check_translation(t, { 0.0f, 0.0f, 0.0f });
    check_scaling(s, { 1.0f, 1.0f, 1.0f });
    check_rotation(r, Rotation<float, 3>{});// identity rotation
}

TEST_CASE("trs_decompose pure translation", "[transform][decompose]") {
    auto T = translation(Vector<float, 3>({ 3.0f, -1.0f, 7.5f }));
    AffineTransform<float> xf = T.to_transform();
    auto [t, r, s] = trs_decompose(xf);

    check_translation(t, { 3.0f, -1.0f, 7.5f });
    check_scaling(s, { 1.0f, 1.0f, 1.0f });
    check_rotation(r, Rotation<float, 3>{});
}

TEST_CASE("trs_decompose pure scaling", "[transform][decompose]") {
    auto S = scaling(Vector<float, 3>({ 2.0f, 3.0f, 0.5f }));
    AffineTransform<float> xf = S.to_transform();
    auto [t, r, s] = trs_decompose(xf);

    check_translation(t, { 0.0f, 0.0f, 0.0f });
    check_scaling(s, { 2.0f, 3.0f, 0.5f });
    // Rotation should be identity (or very close)
    check_rotation(r, Rotation<float, 3>{});
}

TEST_CASE("trs_decompose pure rotation", "[transform][decompose]") {
    auto R = rotation(radians(90.0f), Vector<float, 3>({ 0.0f, 1.0f, 0.0f }));
    AffineTransform<float> xf = R.to_transform();
    auto [t, r, s] = trs_decompose(xf);

    check_translation(t, { 0.0f, 0.0f, 0.0f });
    check_scaling(s, { 1.0f, 1.0f, 1.0f });
    check_rotation(r, R);
}

TEST_CASE("trs_decompose full TRS round-trip", "[transform][decompose]") {
    auto T = translation(Vector<float, 3>({ 1.0f, 2.0f, 3.0f }));
    auto R = rotation(radians(45.0f), Vector<float, 3>({ 0.0f, 1.0f, 0.0f }));
    auto S = scaling(Vector<float, 3>({ 2.0f, 3.0f, 4.0f }));
    AffineTransform<float> xf = T * R * S;

    auto [t, r, s] = trs_decompose(xf);

    check_translation(t, { 1.0f, 2.0f, 3.0f });
    check_scaling(s, { 2.0f, 3.0f, 4.0f });
    check_rotation(r, R);
}

TEST_CASE("trs_decompose uniform scale", "[transform][decompose]") {
    auto T = translation(Vector<float, 3>({ -5.0f, 0.0f, 10.0f }));
    auto R = rotation(radians(30.0f), Vector<float, 3>({ 1.0f, 0.0f, 0.0f }));
    auto S = scaling(Vector<float, 3>({ 5.0f, 5.0f, 5.0f }));
    AffineTransform<float> xf = T * R * S;

    auto [t, r, s] = trs_decompose(xf);

    check_translation(t, { -5.0f, 0.0f, 10.0f });
    check_scaling(s, { 5.0f, 5.0f, 5.0f });
    check_rotation(r, R);
}

TEST_CASE("trs_decompose recomposes to original", "[transform][decompose]") {
    auto T = translation(Vector<float, 3>({ 4.0f, -2.0f, 1.0f }));
    auto R = rotation(radians(60.0f), Vector<float, 3>({ 1.0f, 1.0f, 0.0f }));
    auto S = scaling(Vector<float, 3>({ 1.5f, 2.5f, 0.75f }));
    AffineTransform<float> original = T * R * S;

    auto [t, r, s] = trs_decompose(original);

    // Recompose and compare against original matrix
    AffineTransform<float> recomposed = t * r * s;
    for (index_type row = 0; row < 4; ++row) {
        for (index_type col = 0; col < 4; ++col) {
            CHECK(static_cast<float>(recomposed(row, col)) == Catch::Approx(static_cast<float>(original(row, col))).margin(1e-5f));
        }
    }
}

TEST_CASE("trs_decompose with quaternion round-trip", "[transform][decompose]") {
    // Build from quaternion path
    auto q_in = angle_axis(radians(120.0f), Vector<float, 3>({ 0.0f, 0.0f, 1.0f }));
    auto T = translation(Vector<float, 3>({ 10.0f, 20.0f, 30.0f }));
    auto R_mat = rotation(radians(120.0f), Vector<float, 3>({ 0.0f, 0.0f, 1.0f }));
    auto S = scaling(Vector<float, 3>({ 1.0f, 1.0f, 1.0f }));
    AffineTransform<float> xf = T * R_mat * S;

    auto [t, r, s] = trs_decompose(xf);

    // Convert decomposed rotation to quaternion
    auto q_out = to_quaternion(r.matrix());
    // Quaternions q and -q represent the same rotation, so compare
    // accounting for sign.
    float sign = (q_in.dot(q_out) < 0.0f) ? -1.0f : 1.0f;
    CHECK(static_cast<float>(q_out.w()) * sign == Catch::Approx(static_cast<float>(q_in.w())).margin(1e-5f));
    CHECK(static_cast<float>(q_out.x()) * sign == Catch::Approx(static_cast<float>(q_in.x())).margin(1e-5f));
    CHECK(static_cast<float>(q_out.y()) * sign == Catch::Approx(static_cast<float>(q_in.y())).margin(1e-5f));
    CHECK(static_cast<float>(q_out.z()) * sign == Catch::Approx(static_cast<float>(q_in.z())).margin(1e-5f));
}

TEST_CASE("trs_decompose negative determinant (reflection)", "[transform][decompose]") {
    // Create a transform with a reflection: scale X by -1
    auto S = scaling(Vector<float, 3>({ -2.0f, 3.0f, 4.0f }));
    AffineTransform<float> xf = S.to_transform();

    auto [t, r, s] = trs_decompose(xf);

    check_translation(t, { 0.0f, 0.0f, 0.0f });
    // The first scale factor should be negative
    CHECK(static_cast<float>(s(0)) == Catch::Approx(-2.0f).margin(1e-5f));
    CHECK(static_cast<float>(s(1)) == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(static_cast<float>(s(2)) == Catch::Approx(4.0f).margin(1e-5f));
}

TEST_CASE("trs_decompose isometry input", "[transform][decompose]") {
    // trs_decompose should also work with Isometry (a stricter AffineTransform)
    auto R = rotation(radians(45.0f), Vector<float, 3>({ 0.0f, 0.0f, 1.0f }));
    auto T = translation(Vector<float, 3>({ 1.0f, 2.0f, 3.0f }));
    Isometry<float> xf = T * R;

    auto [t, r, s] = trs_decompose(xf);

    check_translation(t, { 1.0f, 2.0f, 3.0f });
    check_scaling(s, { 1.0f, 1.0f, 1.0f });
    check_rotation(r, R);
}

TEST_CASE("trs_decompose 2D", "[transform][decompose]") {
    // Build a 2D affine transform: translate + rotate + scale
    Translation<float, 2> T({ 5.0f, -3.0f });

    float angle = radians(30.0f);
    Rotation<float, 2> R;
    R(0, 0) = std::cos(angle);
    R(0, 1) = -std::sin(angle);
    R(1, 0) = std::sin(angle);
    R(1, 1) = std::cos(angle);

    Scaling<float, 2> S({ 2.0f, 4.0f });

    AffineTransform<float, 2> xf = T * R * S;
    auto [t, r, s] = trs_decompose(xf);

    CHECK(static_cast<float>(t(0)) == Catch::Approx(5.0f).margin(1e-5f));
    CHECK(static_cast<float>(t(1)) == Catch::Approx(-3.0f).margin(1e-5f));
    CHECK(static_cast<float>(s(0)) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(static_cast<float>(s(1)) == Catch::Approx(4.0f).margin(1e-5f));
    check_rotation(r, R);
}

TEST_CASE("trs_decompose double precision", "[transform][decompose]") {
    auto T = translation(Vector<double, 3>({ 1.0, 2.0, 3.0 }));
    auto R = rotation(radians(45.0), Vector<double, 3>({ 0.0, 1.0, 0.0 }));
    auto S = scaling(Vector<double, 3>({ 2.0, 3.0, 4.0 }));
    AffineTransform<double> xf = T * R * S;

    auto [t, r, s] = trs_decompose(xf);

    CHECK(static_cast<double>(t(0)) == Catch::Approx(1.0).margin(1e-12));
    CHECK(static_cast<double>(t(1)) == Catch::Approx(2.0).margin(1e-12));
    CHECK(static_cast<double>(t(2)) == Catch::Approx(3.0).margin(1e-12));
    CHECK(static_cast<double>(s(0)) == Catch::Approx(2.0).margin(1e-12));
    CHECK(static_cast<double>(s(1)) == Catch::Approx(3.0).margin(1e-12));
    CHECK(static_cast<double>(s(2)) == Catch::Approx(4.0).margin(1e-12));
}
