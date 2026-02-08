#include "catch_include.hpp"

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/MDSpan.hpp>

TEST_CASE("stl_storage_zipper_bases", "[data]") {
  // Vector span wrapping a std::vector via MDSpan (dynamic extent)
  {
    std::vector<double> x = {0, 1, 2};
    using SpanExpr =
        zipper::expression::nullary::MDSpan<double, zipper::extents<zipper::dynamic_extent>>;
    using VSpan = zipper::VectorBase<SpanExpr>;
    VSpan X{std::span<double>(x), zipper::extents<zipper::dynamic_extent>(x.size())};

    CHECK(X(0) == 0);
    CHECK(X(1) == 1);
    CHECK(X(2) == 2);

    CHECK(X.expression().coeff(0) == 0);
    CHECK(X.expression().coeff(1) == 1);
    CHECK(X.expression().coeff(2) == 2);
    CHECK(X.expression().coeff_ref(0) == 0);
    CHECK(X.expression().coeff_ref(1) == 1);
    CHECK(X.expression().coeff_ref(2) == 2);
    CHECK(X.expression().const_coeff_ref(0) == 0);
    CHECK(X.expression().const_coeff_ref(1) == 1);
    CHECK(X.expression().const_coeff_ref(2) == 2);

    X(0) = 4;
    X(1) = 5;
    X(2) = 6;
    CHECK(X(0) == 4);
    CHECK(X(1) == 5);
    CHECK(X(2) == 6);

    using expression_type = std::decay_t<decltype(X)>::expression_type;
    using traits =
        zipper::expression::detail::ExpressionTraits<expression_type>;

    static_assert(!traits::access_features.is_const);
    static_assert(traits::is_writable);
  }

  // Vector span wrapping a std::array via span_type (static extent)
  {
    std::array<double, 3> x{{0, 1, 2}};
    auto sp = std::span<double, 3>(x);
    zipper::Vector<double, 3>::span_type X{sp};

    CHECK(X(0) == 0);
    CHECK(X(1) == 1);
    CHECK(X(2) == 2);

    CHECK(X.expression().coeff(0) == 0);
    CHECK(X.expression().coeff(1) == 1);
    CHECK(X.expression().coeff(2) == 2);
    CHECK(X.expression().coeff_ref(0) == 0);
    CHECK(X.expression().coeff_ref(1) == 1);
    CHECK(X.expression().coeff_ref(2) == 2);
    CHECK(X.expression().const_coeff_ref(0) == 0);
    CHECK(X.expression().const_coeff_ref(1) == 1);
    CHECK(X.expression().const_coeff_ref(2) == 2);

    X(0) = 4;
    X(1) = 5;
    X(2) = 6;

    CHECK(X(0) == 4);
    CHECK(X(1) == 5);
    CHECK(X(2) == 6);

    // Verify data is written through to underlying std::array
    CHECK(x[0] == 4);
    CHECK(x[1] == 5);
    CHECK(x[2] == 6);

    X = {0, 4, 5};
    CHECK(X.size() == 3);
    CHECK(X.rows() == 3);

    CHECK(X(0) == 0);
    CHECK(X(1) == 4);
    CHECK(X(2) == 5);
  }

  // Matrix span wrapping a flat std::array (2x3)
  {
    std::array<double, 6> x{{0, 1, 2, 3, 4, 5}};
    auto sp = std::span<double, 6>(x);
    zipper::Matrix<double, 2, 3>::span_type X{sp};

    CHECK(X(0, 0) == 0);
    CHECK(X(0, 1) == 1);
    CHECK(X(0, 2) == 2);
    CHECK(X(1, 0) == 3);
    CHECK(X(1, 1) == 4);
    CHECK(X(1, 2) == 5);

    CHECK(X.expression().coeff(0, 0) == 0);
    CHECK(X.expression().coeff(0, 1) == 1);
    CHECK(X.expression().coeff(0, 2) == 2);
    CHECK(X.expression().coeff(1, 0) == 3);
    CHECK(X.expression().coeff(1, 1) == 4);
    CHECK(X.expression().coeff(1, 2) == 5);
    CHECK(X.expression().coeff_ref(0, 0) == 0);
    CHECK(X.expression().coeff_ref(0, 1) == 1);
    CHECK(X.expression().coeff_ref(0, 2) == 2);
    CHECK(X.expression().coeff_ref(1, 0) == 3);
    CHECK(X.expression().coeff_ref(1, 1) == 4);
    CHECK(X.expression().coeff_ref(1, 2) == 5);
    CHECK(X.expression().const_coeff_ref(0, 0) == 0);
    CHECK(X.expression().const_coeff_ref(0, 1) == 1);
    CHECK(X.expression().const_coeff_ref(0, 2) == 2);
    CHECK(X.expression().const_coeff_ref(1, 0) == 3);
    CHECK(X.expression().const_coeff_ref(1, 1) == 4);
    CHECK(X.expression().const_coeff_ref(1, 2) == 5);

    X(0, 0) = 14;
    X(0, 1) = 15;
    X(0, 2) = 16;
    X(1, 0) = 24;
    X(1, 1) = 25;
    X(1, 2) = 26;

    CHECK(x[0] == 14);
    CHECK(x[1] == 15);
    CHECK(x[2] == 16);
    CHECK(x[3] == 24);
    CHECK(x[4] == 25);
    CHECK(x[5] == 26);

    CHECK(X(0, 0) == 14);
    CHECK(X(0, 1) == 15);
    CHECK(X(0, 2) == 16);
    CHECK(X(1, 0) == 24);
    CHECK(X(1, 1) == 25);
    CHECK(X(1, 2) == 26);

    zipper::Matrix<double, 2, 3> B;
    CHECK(B.extents() == zipper::create_dextents(2, 3));
    CHECK(X.extents() == zipper::create_dextents(2, 3));
    B.row(0) = {0, 1, 2};
    B.row(1) = {3, 4, 5};
    X = B;
    CHECK(X.extents() == zipper::create_dextents(2, 3));
    CHECK((X == B));
  }
}
