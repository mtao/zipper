#include "catch_include.hpp"

#include <zipper/Form.hpp>
#include <zipper/Vector.hpp>
#include <zipper/exterior.hpp>

TEST_CASE("alternating_projection_zeroes_repeated_indices", "[form][exterior]") {
    zipper::Form<double, 3, 3> F;
    F(0, 1) = 4.0;
    F(1, 0) = -2.0;
    F(2, 2) = 7.0;

    auto A = zipper::alternating_part(F).eval();

    CHECK(A(2, 2) == 0.0);
    CHECK(A(0, 1) == Catch::Approx(3.0));
    CHECK(A(1, 0) == Catch::Approx(-3.0));
}

TEST_CASE("wedge_of_1forms_matches_exterior_definition", "[form][exterior]") {
    zipper::Vector<double, 3> a{{1.0, 2.0, 3.0}};
    zipper::Vector<double, 3> b{{4.0, 5.0, 6.0}};

    auto w = zipper::wedge(a.as_form(), b.as_form()).eval();

    for (zipper::index_type i = 0; i < 3; ++i) {
        for (zipper::index_type j = 0; j < 3; ++j) {
            const double expected = a(i) * b(j) - a(j) * b(i);
            CHECK(w(i, j) == Catch::Approx(expected));
        }
    }
}

TEST_CASE("wedge_of_2form_and_1form_is_alternating", "[form][exterior]") {
    zipper::Form<double, 3, 3> F;
    F(0, 1) = 2.0;
    F(1, 0) = -2.0;
    F(0, 2) = -1.0;
    F(2, 0) = 1.0;
    F(1, 2) = 3.0;
    F(2, 1) = -3.0;
    zipper::Vector<double, 3> v{{7.0, -2.0, 5.0}};

    auto w = zipper::wedge(F, v.as_form()).eval();

    CHECK(w(0, 0, 1) == 0.0);
    CHECK(w(0, 1, 2) == Catch::Approx(F(0, 1) * v(2) - F(0, 2) * v(1) + F(1, 2) * v(0)));
    CHECK(w(1, 0, 2) == Catch::Approx(-w(0, 1, 2)));
    CHECK(w(2, 1, 0) == Catch::Approx(-w(0, 1, 2)));
}

TEST_CASE("basis_coordinates_round_trip_2form", "[form][exterior]") {
    zipper::Form<double, 4, 4> F;
    F(0, 1) = 1.5;
    F(1, 0) = -1.5;
    F(0, 2) = -2.0;
    F(2, 0) = 2.0;
    F(1, 3) = 4.25;
    F(3, 1) = -4.25;

    auto coords = zipper::to_basis_coordinates(F);

    CHECK(coords.extent(0) == 6);
    CHECK(coords(0) == Catch::Approx(1.5));
    CHECK(coords(1) == Catch::Approx(-2.0));
    CHECK(coords(4) == Catch::Approx(4.25));

    auto roundtrip = zipper::from_basis_coordinates<4>(coords, zipper::static_rank_t<2>{});

    for (zipper::index_type i = 0; i < 4; ++i) {
        for (zipper::index_type j = 0; j < 4; ++j) {
            const double expected = zipper::alternating_part(F)(i, j);
            CHECK(roundtrip(i, j) == Catch::Approx(expected));
        }
    }
}

TEST_CASE("hodge_star_on_basis_coordinates_uses_complement_basis", "[form][exterior][hodge]") {
    auto dx = zipper::basis_form<3, 1, double>({0});
    auto star_dx = zipper::hodge_star(dx).eval();

    CHECK(star_dx(1, 2) == Catch::Approx(1.0));
    CHECK(star_dx(2, 1) == Catch::Approx(-1.0));

    auto dx_dy = zipper::wedge(zipper::basis_form<3, 1, double>({0}),
                               zipper::basis_form<3, 1, double>({1})).eval();
    auto star_dx_dy = zipper::hodge_star(dx_dy).eval();

    CHECK(star_dx_dy(2) == Catch::Approx(1.0));
    CHECK(star_dx_dy(0) == Catch::Approx(0.0));
    CHECK(star_dx_dy(1) == Catch::Approx(0.0));
}
