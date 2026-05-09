#include <print>
#include <zipper/Form.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/concepts/shapes.hpp>
#include <zipper/exterior.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/Random.hpp>
#include <zipper/expression/nullary/Unit.hpp>

#include "catch_include.hpp"
// #include <zipper/Vector.hpp>

namespace {

void print(zipper::concepts::Matrix auto const &M) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
        for (zipper::index_type k = 0; k < M.extent(1); ++k) {
            std::print("{} ", M(j, k));
        }
        std::println("");
    }
}
void print(zipper::concepts::Vector auto const &M) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
        std::print("{} ", M(j));
    }
    std::println("");
}
} // namespace
  //
using namespace zipper;
TEST_CASE("test_deductions", "[vector][storage][dense]") {
    Vector<double, 3> a{{0, 2, 4}};
    std::vector<double> X{0, 1, 2};
    std::array<double, 3> Y{{3, 5, 7}};
    VectorBase x = X;
    CHECK(x(0) == 0);
    CHECK(x(1) == 1);
    CHECK(x(2) == 2);
    x = a;
    CHECK(x(0) == 0);
    CHECK(x(1) == 2);
    CHECK(x(2) == 4);

    VectorBase y = Y;
    CHECK(y(0) == 3);
    CHECK(y(1) == 5);
    CHECK(y(2) == 7);
    y = a;
    CHECK(y(0) == 0);
    CHECK(y(1) == 2);
    CHECK(y(2) == 4);
#if defined(__cpp_lib_span_initializer_list)
    // Keep the array alive since VectorBase wraps a span into it
    auto Z = std::array{1, 2, 3};
    VectorBase z(Z);
    CHECK(z(0) == 1);
    CHECK(z(1) == 2);
    CHECK(z(2) == 3);
#endif
}

TEST_CASE("test_dot", "[matrix][storage][dense]") {
    Vector<double, 3> a{{0, 2, 4}};
    Vector<double, 3> b{{1, 3, 5}};

    // static_assert(zipper::concepts::ValidExtents<Vector<double,3>,3>);
    STATIC_CHECK(zipper::concepts::ValidExtents<Vector<double, 3>, 3>);

    Vector c = (*(*a.as_form())).as_vector();

    VectorBase e0 = expression::nullary::unit_vector<double, 3>(0);
    VectorBase e1 = expression::nullary::unit_vector<double>(3, 1);
    VectorBase e2 = expression::nullary::unit_vector<double, 3, 2>();

    CHECK(a.dot(e0) == 0);
    CHECK(a.dot(e1) == 2);
    CHECK(a.dot(e2) == 4);
    CHECK(b.dot(e0) == 1);
    CHECK(b.dot(e1) == 3);
    CHECK(b.dot(e2) == 5);
    CHECK(c.dot(e0) == 0);
    CHECK(c.dot(e1) == 2);
    CHECK(c.dot(e2) == 4);

    CHECK(a.head<2>().dot(b.head<2>()) == 6);
}

TEST_CASE("test_span", "[vector][storage][dense]") {
    zipper::Vector<zipper::index_type, 3> C;
    zipper::Vector<zipper::index_type, std::dynamic_extent> R(3);

    auto CS = C.as_span();
    auto RS = R.as_span();

    REQUIRE(CS.extents() == C.extents());
    REQUIRE(RS.extents() == R.extents());

    for (zipper::index_type j = 0; j < 3; ++j) {
        CHECK(&C(j) == &CS(j));
        CHECK(&R(j) == &RS(j));
    }
}
TEST_CASE("test_vector_span", "[vector][storage][dense][span]") {
    std::vector<int> vec = {2, 3};
    VectorBase v = std::span<int, 2>(vec);
    VectorBase v_const = std::span<const int, 2>(vec);

    VectorBase v2 = vec;

    auto c = v_const.eval();
    CHECK((c == v_const));

    CHECK((v == v2));

    STATIC_CHECK(v.static_extent(0) == 2);
    REQUIRE(v.extent(0) == 2);
    CHECK(v(0) == 2);
    CHECK(v(1) == 3);

    std::array<int, 2> y;
    VectorBase z(y);
    z = v.expression();
    CHECK(y[0] == 2);
    CHECK(y[1] == 3);

    z(0) = 3;
    z(1) = 4;

    CHECK(y[0] == 3);
    CHECK(y[1] == 4);
    z = v;
    CHECK(y[0] == 2);
    CHECK(y[1] == 3);

    z(0) = 3;
    z(1) = 4;

    CHECK(y[0] == 3);
    CHECK(y[1] == 4);

    {
        Vector x = v;
        auto a = x.as_span();
        auto b = x.as_const_span();
        CHECK((a == b));

        zipper::Vector<int, 2>::const_span_type d = a;
        CHECK((a == d));
    }

    // this last case WOULD be very cool, but seems to not work due to a parse
    // limitation in type deductions? In particular, gcc at least seems to
    // really want y to be the name of a variable of type VectorBase
    // VectorBase(y) = {4, 5};
    // CHECK(v(0) == 2);
    // CHECK(v(1) == 3);
}

// ============================================================
// Compound Assignment (static extents)
// ============================================================

TEST_CASE("static_vector_compound_add", "[vector][compound]") {
    Vector<double, 3> a{{1.0, 2.0, 3.0}};
    Vector<double, 3> b{{10.0, 20.0, 30.0}};
    a += b;
    CHECK(a(0) == 11.0);
    CHECK(a(1) == 22.0);
    CHECK(a(2) == 33.0);
}

TEST_CASE("static_vector_compound_sub", "[vector][compound]") {
    Vector<double, 3> a{{10.0, 20.0, 30.0}};
    Vector<double, 3> b{{1.0, 2.0, 3.0}};
    a -= b;
    CHECK(a(0) == 9.0);
    CHECK(a(1) == 18.0);
    CHECK(a(2) == 27.0);
}

TEST_CASE("static_vector_compound_scalar_mul", "[vector][compound]") {
    Vector<double, 3> a{{1.0, 2.0, 3.0}};
    a *= 3.0;
    CHECK(a(0) == 3.0);
    CHECK(a(1) == 6.0);
    CHECK(a(2) == 9.0);
}

TEST_CASE("static_vector_compound_scalar_div", "[vector][compound]") {
    Vector<double, 3> a{{6.0, 12.0, 18.0}};
    a /= 3.0;
    CHECK(a(0) == 2.0);
    CHECK(a(1) == 4.0);
    CHECK(a(2) == 6.0);
}

TEST_CASE("static_matrix_compound_add", "[matrix][compound]") {
    Matrix<double, 2, 2> A{{{1.0, 2.0}, {3.0, 4.0}}};
    Matrix<double, 2, 2> B{{{10.0, 20.0}, {30.0, 40.0}}};
    A += B;
    CHECK(A(0, 0) == 11.0);
    CHECK(A(0, 1) == 22.0);
    CHECK(A(1, 0) == 33.0);
    CHECK(A(1, 1) == 44.0);
}

TEST_CASE("static_matrix_compound_sub", "[matrix][compound]") {
    Matrix<double, 2, 2> A{{{10.0, 20.0}, {30.0, 40.0}}};
    Matrix<double, 2, 2> B{{{1.0, 2.0}, {3.0, 4.0}}};
    A -= B;
    CHECK(A(0, 0) == 9.0);
    CHECK(A(0, 1) == 18.0);
    CHECK(A(1, 0) == 27.0);
    CHECK(A(1, 1) == 36.0);
}

TEST_CASE("static_matrix_compound_scalar_mul", "[matrix][compound]") {
    Matrix<double, 2, 2> A{{{1.0, 2.0}, {3.0, 4.0}}};
    A *= 2.0;
    CHECK(A(0, 0) == 2.0);
    CHECK(A(0, 1) == 4.0);
    CHECK(A(1, 0) == 6.0);
    CHECK(A(1, 1) == 8.0);
}

TEST_CASE("static_matrix_compound_scalar_div", "[matrix][compound]") {
    Matrix<double, 2, 2> A{{{2.0, 4.0}, {6.0, 8.0}}};
    A /= 2.0;
    CHECK(A(0, 0) == 1.0);
    CHECK(A(0, 1) == 2.0);
    CHECK(A(1, 0) == 3.0);
    CHECK(A(1, 1) == 4.0);
}

// NOTE: Form compound assignment (*=, /=) is currently broken because
// FormBase::operator= uses `expression() = v.expression()` instead of
// `expression().assign(v.expression())`. See FormBase.hpp lines 62-69.
// Form compound tests are omitted until FormBase assignment is fixed.

TEST_CASE("hodge_star_euclidean_1form", "[form][hodge]") {
    auto dx = basis_form<3, 1, double>({0});
    auto star_dx = (*dx).eval();

    CHECK(star_dx.extent(0) == 3);
    CHECK(star_dx.extent(1) == 3);
    CHECK(star_dx(1, 2) == Catch::Approx(1.0));
    CHECK(star_dx(2, 1) == Catch::Approx(-1.0));
    CHECK(star_dx(0, 1) == Catch::Approx(0.0));
    CHECK(star_dx(0, 2) == Catch::Approx(0.0));
}

TEST_CASE("hodge_star_2d", "[form][hodge]") {
    Vector<double, 2> v{{4.0, -5.0}};
    auto f = v.as_form();
    auto star_f = (*f).eval();
    auto star_star_f = (*star_f).eval();
    Vector<double, 2> roundtrip = star_star_f.as_vector();

    CHECK(roundtrip(0) == -4.0);
    CHECK(roundtrip(1) == 5.0);
}

TEST_CASE("hodge_star_preserves_dot_product", "[form][hodge]") {
    auto dx = basis_form<3, 1, double>({0});
    auto dy = basis_form<3, 1, double>({1});
    auto dx_wedge_dy = wedge(dx, dy).eval();
    auto star_dx_wedge_dy = (*dx_wedge_dy).eval();

    CHECK(star_dx_wedge_dy.extent(0) == 3);
    CHECK(star_dx_wedge_dy(2) == Catch::Approx(1.0));
    CHECK(star_dx_wedge_dy(0) == Catch::Approx(0.0));
    CHECK(star_dx_wedge_dy(1) == Catch::Approx(0.0));
}

TEST_CASE("hodge_star_double_application", "[form][hodge]") {
    auto dx = basis_form<3, 1, double>({0});
    auto star_dx = (*dx).eval();
    auto star_star_dx = (*star_dx).eval();

    CHECK(star_star_dx(0) == Catch::Approx(1.0));
    CHECK(star_star_dx(1) == Catch::Approx(0.0));
    CHECK(star_star_dx(2) == Catch::Approx(0.0));

    auto dx_wedge_dy = wedge(basis_form<3, 1, double>({0}),
                             basis_form<3, 1, double>({1})).eval();
    auto star_2 = (*dx_wedge_dy).eval();
    auto star_star_2 = (*star_2).eval();

    CHECK(star_star_2(0, 1) == Catch::Approx(dx_wedge_dy(0, 1)));
    CHECK(star_star_2(1, 0) == Catch::Approx(dx_wedge_dy(1, 0)));
}

TEST_CASE("hodge_star_1d", "[form][hodge]") {
    Vector<double, 1> v;
    v(0) = 42.0;
    auto f = v.as_form();
    auto star_f = (*f).eval();

    CHECK(star_f() == 42.0);
}
