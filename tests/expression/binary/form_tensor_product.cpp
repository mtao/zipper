

#include <iostream>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Form.hpp>
#include <zipper/FormBase.hxx>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/Random.hpp>
#include <zipper/expression/nullary/Unit.hpp>

#include "../../catch_include.hpp"

TEST_CASE("test_form_product", "[storage][tensor]") {
    // Test form-tensor contraction (dot product and higher-rank contractions)

    zipper::Vector<double, 3> O;
    O(0) = 1;
    O(1) = 1;
    O(2) = 1;

    zipper::Vector<double, 3> E0 = zipper::expression::nullary::unit_vector<double, 3>(0);
    zipper::Vector<double, 3> E1 = zipper::expression::nullary::unit_vector<double, 3>(1);
    zipper::Vector<double, 3> E2 = zipper::expression::nullary::unit_vector<double, 3>(2);

    zipper::Form<double, 3> D0 = zipper::expression::nullary::unit_vector<double, 3>(0);
    zipper::Form<double, 3> D1 = zipper::expression::nullary::unit_vector<double, 3>(1);
    zipper::Form<double, 3> D2 = zipper::expression::nullary::unit_vector<double, 3>(2);

    // Test that form * vector gives the Kronecker delta (identity)
    CHECK(double(D0 * E0) == 1.0);
    CHECK(double(D1 * E0) == 0.0);
    CHECK(double(D2 * E0) == 0.0);

    CHECK(double(D0 * E1) == 0.0);
    CHECK(double(D1 * E1) == 1.0);
    CHECK(double(D2 * E1) == 0.0);

    CHECK(double(D0 * E2) == 0.0);
    CHECK(double(D1 * E2) == 0.0);
    CHECK(double(D2 * E2) == 1.0);

    // Test dot product: form(O) * O = sum of components squared
    CHECK(double(O.as_form() * O) == Catch::Approx(3.0));

    // Test rank-2 form * rank-1 tensor contraction → rank-1 form
    // FormTensorProduct requires form_rank >= tensor_rank
    // (F * v)(i) = sum_j F(i,j) * v(j)
    zipper::Vector<double, 3> a =
        zipper::expression::nullary::normal_random_infinite<double>(0, 1);
    zipper::Vector<double, 3> b =
        zipper::expression::nullary::normal_random_infinite<double>(0, 1);

    zipper::Vector<double, 3> x =
        zipper::expression::nullary::normal_random_infinite<double>(0, 1);
    zipper::Vector<double, 3> y =
        zipper::expression::nullary::normal_random_infinite<double>(0, 1);

    // Build a rank-2 form from wedge product: F = x^ ^ y^
    auto F_xy = (x.as_form() ^ y.as_form()).eval();
    static_assert(decltype(F_xy)::extents_type::rank() == 2);

    // Test wedge product antisymmetry: (x^ ^ y^)(i,j) = x(i)*y(j) - x(j)*y(i)
    for (zipper::index_type i = 0; i < 3; ++i) {
        for (zipper::index_type j = 0; j < 3; ++j) {
            double expected_val = x(i) * y(j) - x(j) * y(i);
            CHECK(F_xy(i, j) == Catch::Approx(expected_val));
            // Antisymmetry: w(i,j) = -w(j,i)
            CHECK(F_xy(i, j) == Catch::Approx(-F_xy(j, i)));
        }
    }

    // Contract rank-2 form with rank-1 tensor: result is rank-1 form
    // (F_xy * a)(i) = sum_j F_xy(i,j) * a(j)
    auto F_xy_a = (F_xy * a).eval();
    static_assert(decltype(F_xy_a)::extents_type::rank() == 1);

    for (zipper::index_type i = 0; i < 3; ++i) {
        double expected_val = 0.0;
        for (zipper::index_type j = 0; j < 3; ++j) {
            expected_val += F_xy(i, j) * a(j);
        }
        CHECK(F_xy_a(i) == Catch::Approx(expected_val));
    }

    // Chained contraction: (F_xy * a) * b should give scalar
    // sum_i (sum_j F_xy(i,j) * a(j)) * b(i)
    double chained = double(F_xy_a * b);
    double expected_chained = 0.0;
    for (zipper::index_type i = 0; i < 3; ++i) {
        for (zipper::index_type j = 0; j < 3; ++j) {
            expected_chained += F_xy(i, j) * a(j) * b(i);
        }
    }
    CHECK(chained == Catch::Approx(expected_chained));

    // Test rank-1 form * rank-2 tensor → rank-1 tensor
    // When form_rank < tensor_rank, result is a TensorBase
    // (f * T)(j) = sum_i f(i) * T(i, j)
    auto AB = (a.as_tensor() * b.as_tensor()).eval();
    static_assert(decltype(AB)::extents_type::rank() == 2);

    auto xf = x.as_form();
    auto xf_AB = xf * AB;
    static_assert(std::decay_t<decltype(xf_AB)>::extents_type::rank() == 1);

    // (x^ * (a⊗b))(j) = sum_i x(i) * a(i) * b(j) = (x·a) * b(j)
    double x_dot_a = double(x.as_form() * a);
    for (zipper::index_type j = 0; j < 3; ++j) {
        CHECK(double(xf_AB(j)) == Catch::Approx(x_dot_a * b(j)));
    }
}
