#include <cmath>
#include <numbers>

#include <zipper/Matrix.hpp>
#include <zipper/Quaternion.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/transform/quaternion_transform.hpp>
#include <zipper/transform/common.hpp>

#include "../catch_include.hpp"

using namespace zipper;
using namespace zipper::transform;

TEST_CASE("angle_axis identity", "[transform][quaternion_transform]") {
    Vector<float, 3> axis({0.0f, 1.0f, 0.0f});
    auto q = angle_axis(0.0f, axis);
    CHECK(q.w() == Catch::Approx(1.0f));
    CHECK(q.x() == Catch::Approx(0.0f).margin(1e-7f));
    CHECK(q.y() == Catch::Approx(0.0f).margin(1e-7f));
    CHECK(q.z() == Catch::Approx(0.0f).margin(1e-7f));
}

TEST_CASE("angle_axis 90 degrees Y", "[transform][quaternion_transform]") {
    Vector<float, 3> axis({0.0f, 1.0f, 0.0f});
    auto q = angle_axis(radians(90.0f), axis);

    float expected_w = std::cos(radians(90.0f) / 2.0f);
    float expected_y = std::sin(radians(90.0f) / 2.0f);

    CHECK(q.w() == Catch::Approx(expected_w));
    CHECK(q.x() == Catch::Approx(0.0f).margin(1e-7f));
    CHECK(q.y() == Catch::Approx(expected_y));
    CHECK(q.z() == Catch::Approx(0.0f).margin(1e-7f));
}

TEST_CASE("angle_axis unit quaternion", "[transform][quaternion_transform]") {
    Vector<float, 3> axis({1.0f, 0.0f, 0.0f});
    auto q = angle_axis(radians(45.0f), axis);
    CHECK(q.norm() == Catch::Approx(1.0f));
}

TEST_CASE("to_rotation_matrix identity quaternion", "[transform][quaternion_transform]") {
    Quaternion<float> q(1.0f, 0.0f, 0.0f, 0.0f);
    auto M = to_rotation_matrix(q);

    for (index_type r = 0; r < 3; ++r) {
        for (index_type c = 0; c < 3; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(M(r, c) == Catch::Approx(expected).margin(1e-6f));
        }
    }
}

TEST_CASE("to_affine identity quaternion", "[transform][quaternion_transform]") {
    Quaternion<float> q(1.0f, 0.0f, 0.0f, 0.0f);
    auto A = to_affine(q);

    for (index_type r = 0; r < 4; ++r) {
        for (index_type c = 0; c < 4; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(A(r, c) == Catch::Approx(expected).margin(1e-6f));
        }
    }
}

TEST_CASE("to_rotation_matrix 90deg Z rotation", "[transform][quaternion_transform]") {
    Vector<float, 3> axis({0.0f, 0.0f, 1.0f});
    auto q = angle_axis(radians(90.0f), axis);
    auto M = to_rotation_matrix(q);

    // Rotating (1,0,0) by 90 about Z should give (0,1,0)
    CHECK(M(0, 0) == Catch::Approx(0.0f).margin(1e-6f));
    CHECK(M(1, 0) == Catch::Approx(1.0f).margin(1e-6f));
    CHECK(M(2, 0) == Catch::Approx(0.0f).margin(1e-6f));
}

TEST_CASE("to_quaternion roundtrip 3x3", "[transform][quaternion_transform]") {
    Vector<float, 3> axis({0.0f, 1.0f, 0.0f});
    auto q1 = angle_axis(radians(60.0f), axis);
    auto M = to_rotation_matrix(q1);
    auto q2 = to_quaternion(M);

    // q1 and q2 should be the same quaternion (or negated — both represent
    // the same rotation)
    float dot = q1.dot(q2);
    CHECK(std::abs(dot) == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("to_quaternion roundtrip affine", "[transform][quaternion_transform]") {
    Vector<float, 3> axis({1.0f, 0.0f, 0.0f});
    auto q1 = angle_axis(radians(120.0f), axis);
    auto A = to_affine(q1);
    auto q2 = to_quaternion(A);

    float dot = q1.dot(q2);
    CHECK(std::abs(dot) == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("to_rotation_matrix then to_quaternion is rotation-preserving", "[transform][quaternion_transform]") {
    // A rotation should map a vector the same way through both representations
    Vector<float, 3> axis({0.577350269f, 0.577350269f, 0.577350269f}); // normalized (1,1,1)
    auto q = angle_axis(radians(120.0f), axis);
    auto M = to_rotation_matrix(q);

    // Apply rotation via matrix to (1, 0, 0)
    Vector<float, 3> v({1.0f, 0.0f, 0.0f});
    Vector<float, 3> rotated = Vector<float, 3>(M * v);

    // 120 around (1,1,1)/sqrt(3) maps (1,0,0) -> (0,1,0)
    CHECK(rotated(0) == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(rotated(1) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(rotated(2) == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("euler_angles identity", "[transform][quaternion_transform]") {
    Quaternion<float> q(1.0f, 0.0f, 0.0f, 0.0f);
    auto angles = euler_angles(q);
    CHECK(angles(0) == Catch::Approx(0.0f).margin(1e-6f));
    CHECK(angles(1) == Catch::Approx(0.0f).margin(1e-6f));
    CHECK(angles(2) == Catch::Approx(0.0f).margin(1e-6f));
}

TEST_CASE("slerp endpoints", "[transform][quaternion_transform]") {
    Quaternion<float> q1(1.0f, 0.0f, 0.0f, 0.0f);
    Vector<float, 3> axis({0.0f, 1.0f, 0.0f});
    auto q2 = angle_axis(radians(90.0f), axis);

    auto start = slerp(q1, q2, 0.0f);
    CHECK(start.w() == Catch::Approx(q1.w()).margin(1e-6f));
    CHECK(start.x() == Catch::Approx(q1.x()).margin(1e-6f));
    CHECK(start.y() == Catch::Approx(q1.y()).margin(1e-6f));
    CHECK(start.z() == Catch::Approx(q1.z()).margin(1e-6f));

    auto end = slerp(q1, q2, 1.0f);
    CHECK(end.w() == Catch::Approx(q2.w()).margin(1e-5f));
    CHECK(end.x() == Catch::Approx(q2.x()).margin(1e-5f));
    CHECK(end.y() == Catch::Approx(q2.y()).margin(1e-5f));
    CHECK(end.z() == Catch::Approx(q2.z()).margin(1e-5f));
}

TEST_CASE("slerp midpoint", "[transform][quaternion_transform]") {
    Quaternion<float> q1(1.0f, 0.0f, 0.0f, 0.0f);
    Vector<float, 3> axis({0.0f, 1.0f, 0.0f});
    auto q2 = angle_axis(radians(90.0f), axis);

    auto mid = slerp(q1, q2, 0.5f);
    // Midpoint should be a 45 degree rotation around Y
    auto expected = angle_axis(radians(45.0f), axis);

    CHECK(mid.w() == Catch::Approx(expected.w()).margin(1e-5f));
    CHECK(mid.x() == Catch::Approx(expected.x()).margin(1e-5f));
    CHECK(mid.y() == Catch::Approx(expected.y()).margin(1e-5f));
    CHECK(mid.z() == Catch::Approx(expected.z()).margin(1e-5f));
}

TEST_CASE("slerp unit quaternion", "[transform][quaternion_transform]") {
    Quaternion<float> q1(1.0f, 0.0f, 0.0f, 0.0f);
    Vector<float, 3> axis({0.0f, 0.0f, 1.0f});
    auto q2 = angle_axis(radians(180.0f), axis);

    for (float t = 0.0f; t <= 1.0f; t += 0.1f) {
        auto q = slerp(q1, q2, t);
        CHECK(q.norm() == Catch::Approx(1.0f).margin(1e-5f));
    }
}

TEST_CASE("slerp opposite quaternions", "[transform][quaternion_transform]") {
    // q and -q represent the same rotation; slerp should handle this
    Quaternion<double> q1(1.0, 0.0, 0.0, 0.0);
    Quaternion<double> q2(-1.0, 0.0, 0.0, 0.0);  // same rotation, negated

    auto mid = slerp(q1, q2, 0.5);
    CHECK(mid.norm() == Catch::Approx(1.0).margin(1e-10));
}
