#include <cmath>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/transform/view.hpp>

#include "../catch_include.hpp"

using namespace zipper;
using namespace zipper::transform;

TEST_CASE("look_at identity case", "[transform][view]") {
    // Camera at origin, looking down -Z, up = Y
    Vector<float, 3> eye({0.0f, 0.0f, 0.0f});
    Vector<float, 3> center({0.0f, 0.0f, -1.0f});
    Vector<float, 3> up({0.0f, 1.0f, 0.0f});

    auto V = look_at(eye, center, up);

    // Should produce identity (camera already at origin, looking down -Z)
    for (index_type r = 0; r < 4; ++r) {
        for (index_type c = 0; c < 4; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            CHECK(V(r, c) == Catch::Approx(expected).margin(1e-6f));
        }
    }
}

TEST_CASE("look_at translation", "[transform][view]") {
    // Camera at (0, 0, 5), looking at origin
    Vector<float, 3> eye({0.0f, 0.0f, 5.0f});
    Vector<float, 3> center({0.0f, 0.0f, 0.0f});
    Vector<float, 3> up({0.0f, 1.0f, 0.0f});

    auto V = look_at(eye, center, up);

    // The origin should be at (0, 0, -5) in view space
    Vector<float, 4> origin({0.0f, 0.0f, 0.0f, 1.0f});
    Vector<float, 4> result = Vector<float, 4>(V * origin);

    CHECK(result(0) == Catch::Approx(0.0f).margin(1e-6f));
    CHECK(result(1) == Catch::Approx(0.0f).margin(1e-6f));
    CHECK(result(2) == Catch::Approx(-5.0f).margin(1e-6f));
    CHECK(result(3) == Catch::Approx(1.0f).margin(1e-6f));
}

TEST_CASE("look_at right-handedness", "[transform][view]") {
    // The view matrix should be right-handed:
    // determinant of the upper-left 3x3 should be +1
    Vector<float, 3> eye({1.0f, 2.0f, 3.0f});
    Vector<float, 3> center({0.0f, 0.0f, 0.0f});
    Vector<float, 3> up({0.0f, 1.0f, 0.0f});

    auto V = look_at(eye, center, up);

    // Extract 3x3 rotation block
    float det = V(0, 0) * (V(1, 1) * V(2, 2) - V(1, 2) * V(2, 1))
              - V(0, 1) * (V(1, 0) * V(2, 2) - V(1, 2) * V(2, 0))
              + V(0, 2) * (V(1, 0) * V(2, 1) - V(1, 1) * V(2, 0));

    CHECK(det == Catch::Approx(1.0f).margin(1e-5f));
}
