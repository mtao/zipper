#include <cmath>

#include <zipper/Quaternion.hpp>
#include <zipper/Vector.hpp>

#include "../catch_include.hpp"

using namespace zipper;

TEST_CASE("quaternion default construction", "[quaternion]") {
    Quaternion<float> q;
    // Default should be zero-initialized
    CHECK(q.w() == Catch::Approx(0.0f));
    CHECK(q.x() == Catch::Approx(0.0f));
    CHECK(q.y() == Catch::Approx(0.0f));
    CHECK(q.z() == Catch::Approx(0.0f));
}

TEST_CASE("quaternion component construction", "[quaternion]") {
    Quaternion<float> q(1.0f, 2.0f, 3.0f, 4.0f);
    CHECK(q.w() == Catch::Approx(1.0f));
    CHECK(q.x() == Catch::Approx(2.0f));
    CHECK(q.y() == Catch::Approx(3.0f));
    CHECK(q.z() == Catch::Approx(4.0f));
}

TEST_CASE("quaternion initializer list", "[quaternion]") {
    Quaternion<float> q({1.0f, 0.0f, 0.0f, 0.0f});
    CHECK(q.w() == Catch::Approx(1.0f));
    CHECK(q.x() == Catch::Approx(0.0f));
    CHECK(q.y() == Catch::Approx(0.0f));
    CHECK(q.z() == Catch::Approx(0.0f));
}

TEST_CASE("quaternion norm", "[quaternion]") {
    Quaternion<double> q(1.0, 0.0, 0.0, 0.0);
    CHECK(q.norm() == Catch::Approx(1.0));

    Quaternion<double> q2(1.0, 1.0, 1.0, 1.0);
    CHECK(q2.norm() == Catch::Approx(2.0));
}

TEST_CASE("quaternion squaredNorm", "[quaternion]") {
    Quaternion<double> q(1.0, 2.0, 3.0, 4.0);
    CHECK(q.squaredNorm() == Catch::Approx(30.0));
}

TEST_CASE("quaternion normalized", "[quaternion]") {
    Quaternion<double> q(1.0, 1.0, 1.0, 1.0);
    auto n = q.normalized();
    CHECK(n.norm() == Catch::Approx(1.0));
    CHECK(n.w() == Catch::Approx(0.5));
    CHECK(n.x() == Catch::Approx(0.5));
    CHECK(n.y() == Catch::Approx(0.5));
    CHECK(n.z() == Catch::Approx(0.5));
}

TEST_CASE("quaternion normalize in-place", "[quaternion]") {
    Quaternion<double> q(3.0, 0.0, 0.0, 0.0);
    q.normalize();
    CHECK(q.norm() == Catch::Approx(1.0));
    CHECK(q.w() == Catch::Approx(1.0));
}

TEST_CASE("quaternion conjugate", "[quaternion]") {
    Quaternion<double> q(1.0, 2.0, 3.0, 4.0);
    auto c = q.conjugate();
    CHECK(c.w() == Catch::Approx(1.0));
    CHECK(c.x() == Catch::Approx(-2.0));
    CHECK(c.y() == Catch::Approx(-3.0));
    CHECK(c.z() == Catch::Approx(-4.0));
}

TEST_CASE("quaternion inverse", "[quaternion]") {
    Quaternion<double> q(1.0, 0.0, 0.0, 0.0);
    auto inv = q.inverse();
    CHECK(inv.w() == Catch::Approx(1.0));
    CHECK(inv.x() == Catch::Approx(0.0));
    CHECK(inv.y() == Catch::Approx(0.0));
    CHECK(inv.z() == Catch::Approx(0.0));
}

TEST_CASE("quaternion inverse general", "[quaternion]") {
    Quaternion<double> q(1.0, 2.0, 3.0, 4.0);
    auto inv = q.inverse();
    // q * q^-1 should be identity
    Quaternion<double> product = (q * inv).eval();
    CHECK(product.w() == Catch::Approx(1.0).margin(1e-10));
    CHECK(product.x() == Catch::Approx(0.0).margin(1e-10));
    CHECK(product.y() == Catch::Approx(0.0).margin(1e-10));
    CHECK(product.z() == Catch::Approx(0.0).margin(1e-10));
}

TEST_CASE("quaternion dot", "[quaternion]") {
    Quaternion<double> q1(1.0, 0.0, 0.0, 0.0);
    Quaternion<double> q2(0.0, 1.0, 0.0, 0.0);
    CHECK(q1.dot(q2) == Catch::Approx(0.0));

    CHECK(q1.dot(q1) == Catch::Approx(1.0));
}

TEST_CASE("quaternion hamilton product identity", "[quaternion]") {
    Quaternion<double> identity(1.0, 0.0, 0.0, 0.0);
    Quaternion<double> q(0.5, 0.5, 0.5, 0.5);

    // identity * q = q
    Quaternion<double> result = (identity * q).eval();
    CHECK(result.w() == Catch::Approx(q.w()));
    CHECK(result.x() == Catch::Approx(q.x()));
    CHECK(result.y() == Catch::Approx(q.y()));
    CHECK(result.z() == Catch::Approx(q.z()));
}

TEST_CASE("quaternion hamilton product lazy evaluation", "[quaternion]") {
    Quaternion<double> q1(1.0, 0.0, 0.0, 0.0);
    Quaternion<double> q2(0.0, 1.0, 0.0, 0.0);

    // The product should be lazy — accessing individual components
    auto product = q1 * q2;
    CHECK(product.w() == Catch::Approx(0.0));
    CHECK(product.x() == Catch::Approx(1.0));
    CHECK(product.y() == Catch::Approx(0.0));
    CHECK(product.z() == Catch::Approx(0.0));
}

TEST_CASE("quaternion hamilton product i*j=k", "[quaternion]") {
    Quaternion<double> i(0.0, 1.0, 0.0, 0.0);
    Quaternion<double> j(0.0, 0.0, 1.0, 0.0);

    // i * j = k
    Quaternion<double> result = (i * j).eval();
    CHECK(result.w() == Catch::Approx(0.0));
    CHECK(result.x() == Catch::Approx(0.0));
    CHECK(result.y() == Catch::Approx(0.0));
    CHECK(result.z() == Catch::Approx(1.0));
}

TEST_CASE("quaternion hamilton product j*k=i", "[quaternion]") {
    Quaternion<double> j(0.0, 0.0, 1.0, 0.0);
    Quaternion<double> k(0.0, 0.0, 0.0, 1.0);

    Quaternion<double> result = (j * k).eval();
    CHECK(result.w() == Catch::Approx(0.0));
    CHECK(result.x() == Catch::Approx(1.0));
    CHECK(result.y() == Catch::Approx(0.0));
    CHECK(result.z() == Catch::Approx(0.0));
}

TEST_CASE("quaternion hamilton product k*i=j", "[quaternion]") {
    Quaternion<double> k(0.0, 0.0, 0.0, 1.0);
    Quaternion<double> i(0.0, 1.0, 0.0, 0.0);

    Quaternion<double> result = (k * i).eval();
    CHECK(result.w() == Catch::Approx(0.0));
    CHECK(result.x() == Catch::Approx(0.0));
    CHECK(result.y() == Catch::Approx(1.0));
    CHECK(result.z() == Catch::Approx(0.0));
}

TEST_CASE("quaternion scalar multiply", "[quaternion]") {
    Quaternion<double> q(1.0, 2.0, 3.0, 4.0);
    auto r = q * 2.0;
    CHECK(r.w() == Catch::Approx(2.0));
    CHECK(r.x() == Catch::Approx(4.0));
    CHECK(r.y() == Catch::Approx(6.0));
    CHECK(r.z() == Catch::Approx(8.0));

    auto r2 = 3.0 * q;
    CHECK(r2.w() == Catch::Approx(3.0));
    CHECK(r2.x() == Catch::Approx(6.0));
    CHECK(r2.y() == Catch::Approx(9.0));
    CHECK(r2.z() == Catch::Approx(12.0));
}

TEST_CASE("quaternion scalar divide", "[quaternion]") {
    Quaternion<double> q(2.0, 4.0, 6.0, 8.0);
    auto r = q / 2.0;
    CHECK(r.w() == Catch::Approx(1.0));
    CHECK(r.x() == Catch::Approx(2.0));
    CHECK(r.y() == Catch::Approx(3.0));
    CHECK(r.z() == Catch::Approx(4.0));
}

TEST_CASE("quaternion addition", "[quaternion]") {
    Quaternion<double> a(1.0, 2.0, 3.0, 4.0);
    Quaternion<double> b(5.0, 6.0, 7.0, 8.0);
    auto r = a + b;
    CHECK(r.w() == Catch::Approx(6.0));
    CHECK(r.x() == Catch::Approx(8.0));
    CHECK(r.y() == Catch::Approx(10.0));
    CHECK(r.z() == Catch::Approx(12.0));
}

TEST_CASE("quaternion subtraction", "[quaternion]") {
    Quaternion<double> a(5.0, 6.0, 7.0, 8.0);
    Quaternion<double> b(1.0, 2.0, 3.0, 4.0);
    auto r = a - b;
    CHECK(r.w() == Catch::Approx(4.0));
    CHECK(r.x() == Catch::Approx(4.0));
    CHECK(r.y() == Catch::Approx(4.0));
    CHECK(r.z() == Catch::Approx(4.0));
}

TEST_CASE("quaternion negation", "[quaternion]") {
    Quaternion<double> q(1.0, 2.0, 3.0, 4.0);
    auto r = -q;
    CHECK(r.w() == Catch::Approx(-1.0));
    CHECK(r.x() == Catch::Approx(-2.0));
    CHECK(r.y() == Catch::Approx(-3.0));
    CHECK(r.z() == Catch::Approx(-4.0));
}

TEST_CASE("quaternion equality", "[quaternion]") {
    Quaternion<double> a(1.0, 2.0, 3.0, 4.0);
    Quaternion<double> b(1.0, 2.0, 3.0, 4.0);
    Quaternion<double> c(1.0, 2.0, 3.0, 5.0);
    CHECK(a == b);
    CHECK(a != c);
}

TEST_CASE("quaternion as_vector", "[quaternion]") {
    Quaternion<double> q(1.0, 2.0, 3.0, 4.0);
    auto v = q.as_vector();
    CHECK(v(0) == Catch::Approx(1.0));
    CHECK(v(1) == Catch::Approx(2.0));
    CHECK(v(2) == Catch::Approx(3.0));
    CHECK(v(3) == Catch::Approx(4.0));
}
