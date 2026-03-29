#include <cmath>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/transform/coordinate.hpp>
#include <zipper/transform/projection.hpp>
#include <zipper/transform/common.hpp>

#include "../catch_include.hpp"

using namespace zipper;
using namespace zipper::transform;

TEST_CASE("project/unproject roundtrip", "[transform][coordinate]") {
    float fovy = radians(45.0f);
    auto proj = perspective(fovy, 1.0f, 0.1f, 100.0f);
    Isometry<float, 3> model;  // identity
    Vector<float, 4> viewport({0.0f, 0.0f, 800.0f, 600.0f});

    Vector<float, 3> obj({1.0f, 2.0f, -5.0f});
    auto win = project(obj, model, proj, viewport);
    auto obj2 = unproject(win, model, proj, viewport);

    CHECK(obj2(0) == Catch::Approx(obj(0)).margin(1e-4f));
    CHECK(obj2(1) == Catch::Approx(obj(1)).margin(1e-4f));
    CHECK(obj2(2) == Catch::Approx(obj(2)).margin(1e-4f));
}

TEST_CASE("project origin at center", "[transform][coordinate]") {
    auto proj = perspective(radians(90.0f), 1.0f, 1.0f, 100.0f);
    Isometry<float, 3> model;  // identity
    Vector<float, 4> viewport({0.0f, 0.0f, 100.0f, 100.0f});

    // A point on the -Z axis should map to the center of the viewport
    Vector<float, 3> obj({0.0f, 0.0f, -10.0f});
    auto win = project(obj, model, proj, viewport);

    CHECK(win(0) == Catch::Approx(50.0f).margin(1e-3f));
    CHECK(win(1) == Catch::Approx(50.0f).margin(1e-3f));
}
