#include <cmath>

#include <zipper/Vector.hpp>
#include <zipper/transform/geometric.hpp>

#include "../catch_include.hpp"

using namespace zipper;
using namespace zipper::transform;

TEST_CASE("distance", "[transform][geometric]") {
    Vector<float, 3> a({0.0f, 0.0f, 0.0f});
    Vector<float, 3> b({3.0f, 4.0f, 0.0f});
    CHECK(distance(a, b) == Catch::Approx(5.0f));
}

TEST_CASE("distance same point", "[transform][geometric]") {
    Vector<float, 3> a({1.0f, 2.0f, 3.0f});
    CHECK(distance(a, a) == Catch::Approx(0.0f));
}

TEST_CASE("reflect", "[transform][geometric]") {
    // Reflect (1, -1, 0) around normal (0, 1, 0)
    Vector<float, 3> I({1.0f, -1.0f, 0.0f});
    Vector<float, 3> N({0.0f, 1.0f, 0.0f});

    auto R = reflect(I, N);
    CHECK(R(0) == Catch::Approx(1.0f));
    CHECK(R(1) == Catch::Approx(1.0f));
    CHECK(R(2) == Catch::Approx(0.0f));
}

TEST_CASE("reflect perpendicular", "[transform][geometric]") {
    // Reflecting a vector perpendicular to the normal should leave it unchanged
    Vector<float, 3> I({1.0f, 0.0f, 0.0f});
    Vector<float, 3> N({0.0f, 1.0f, 0.0f});

    auto R = reflect(I, N);
    CHECK(R(0) == Catch::Approx(1.0f));
    CHECK(R(1) == Catch::Approx(0.0f));
    CHECK(R(2) == Catch::Approx(0.0f));
}

TEST_CASE("refract no total internal reflection", "[transform][geometric]") {
    Vector<float, 3> I({0.0f, -1.0f, 0.0f});
    Vector<float, 3> N({0.0f, 1.0f, 0.0f});
    float eta = 1.0f;  // Same medium, no refraction

    auto R = refract(I, N, eta);
    CHECK(R(0) == Catch::Approx(0.0f).margin(1e-6f));
    CHECK(R(1) == Catch::Approx(-1.0f).margin(1e-6f));
    CHECK(R(2) == Catch::Approx(0.0f).margin(1e-6f));
}

TEST_CASE("face_forward", "[transform][geometric]") {
    Vector<float, 3> N({0.0f, 1.0f, 0.0f});
    Vector<float, 3> I({0.0f, -1.0f, 0.0f});
    Vector<float, 3> Nref({0.0f, 1.0f, 0.0f});

    // dot(Nref, I) < 0, so N is returned
    auto result = face_forward(N, I, Nref);
    CHECK(result(0) == Catch::Approx(0.0f));
    CHECK(result(1) == Catch::Approx(1.0f));
    CHECK(result(2) == Catch::Approx(0.0f));

    // dot(Nref, I) > 0, so -N is returned
    Vector<float, 3> I2({0.0f, 1.0f, 0.0f});
    auto result2 = face_forward(N, I2, Nref);
    CHECK(result2(0) == Catch::Approx(0.0f));
    CHECK(result2(1) == Catch::Approx(-1.0f));
    CHECK(result2(2) == Catch::Approx(0.0f));
}
