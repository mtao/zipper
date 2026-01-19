
#include <iostream>
#include <zipper/concepts/Matrix.hpp>
#include <zipper/concepts/Tensor.hpp>
#include <zipper/concepts/Vector.hpp>

#include "../../catch_include.hpp"
#include "../../fmt_include.hpp"
// #include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/Random.hpp>
// #include <zipper/expression/nullary/Unit.hpp>
// #include <zipper/expression/reductions/Determinant.hpp>
#include <zipper/expression/reductions/CoefficientSum.hpp>
#include <zipper/expression/reductions/Trace.hpp>
#include <zipper/expression/unary/PartialReduction.hpp>
#include <zipper/expression/unary/detail/PartialReductionDispatcher.hpp>
#include <zipper/utils/format.hpp>

// #include <zipper/Vector.hpp>

namespace {
template <typename T> void print2(T const &M) {
  if constexpr (T::extents_type::rank() == 1) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
      std::cout << M(j) << " ";
    }
    std::cout << std::endl;
  } else if constexpr (T::extents_type::rank() == 2) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
      for (zipper::index_type k = 0; k < M.extent(1); ++k) {
        std::cout << M(j, k) << " ";
      }
      std::cout << std::endl;
    }
  }
}

void print(zipper::concepts::MatrixExpression auto const &M) {
  fmt::print("{}\n", M);
  for (zipper::index_type j = 0; j < M.extent(0); ++j) {
    for (zipper::index_type k = 0; k < M.extent(1); ++k) {
      std::cout << M(j, k) << " ";
    }
    std::cout << std::endl;
  }
}
void print(zipper::concepts::VectorBaseDerived auto const &M) {
  fmt::print("{}\n", M);
  for (zipper::index_type j = 0; j < M.extent(0); ++j) {
    std::cout << M(j) << " ";
  }
  std::cout << std::endl;
}
} // namespace
  //
using namespace zipper;

namespace {
template <typename Type, int index>
auto make_sum(const Type &a, std::integral_constant<int, index>) {
  // return expression::unary::PartialReduction<
  //     Type, expression::reductions::CoefficientSum, index>(a);
  return expression::unary::detail::PartialReductionDispatcher<const Type,
                                                               index>(a)
      .sum();
}
template <typename Type, int index>
auto make_norm(const Type &a, std::integral_constant<int, index>) {
  // return expression::unary::PartialReduction<
  //     Type, expression::reductions::CoefficientSum, index>(a);
  return expression::unary::detail::PartialReductionDispatcher<const Type,
                                                               index>(a)
      .norm();
}
} // namespace

TEST_CASE("test_partial_sum", "[expression][unary]") {
  {
    using IType = expression::nullary::Identity<double, 3, 3>;
    IType i;

    auto pr = make_sum(i, std::integral_constant<int, 1>{});

    REQUIRE(pr.extents().rank() == 1);
    REQUIRE(pr.static_extent(0) == 3);
    REQUIRE(pr.extent(0) == 3);

    CHECK(pr(0) == 1);
    CHECK(pr(1) == 1);
    CHECK(pr(2) == 1);
  }
  {
    Matrix<double, 3, 4> A = expression::nullary::Identity<double, 3, 4>();

    auto pr = make_sum(A.view(), std::integral_constant<int, 1>{});
    auto prn = make_norm(A.view(), std::integral_constant<int, 1>{});
    print2(pr);
    print2(prn);

    REQUIRE(pr.extents().rank() == 1);
    REQUIRE(pr.static_extent(0) == 3);
    REQUIRE(pr.extent(0) == 3);
    REQUIRE(prn.extents().rank() == 1);
    REQUIRE(prn.static_extent(0) == 3);
    REQUIRE(prn.extent(0) == 3);

    auto pr2 = make_sum(A.view(), std::integral_constant<int, 0>{});
    auto pr2n = make_norm(A.view(), std::integral_constant<int, 0>{});

    REQUIRE(pr2.extents().rank() == 1);
    REQUIRE(pr2.static_extent(0) == 4);
    REQUIRE(pr2.extent(0) == 4);
    REQUIRE(pr2n.extents().rank() == 1);
    REQUIRE(pr2n.static_extent(0) == 4);
    REQUIRE(pr2n.extent(0) == 4);

    // slice of rows
    CHECK(pr(0) == 1);
    CHECK(pr(1) == 1);
    CHECK(pr(2) == 1);

    // slice of cols
    CHECK(pr2(0) == 1);
    CHECK(pr2(1) == 1);
    CHECK(pr2(2) == 1);
    CHECK(pr2(3) == 0);

    // update a value
    A(0, 3) = 10;
    // slice of rows
    CHECK(pr(0) == 11);
    CHECK(pr(1) == 1);
    CHECK(pr(2) == 1);

    // slice of cols
    CHECK(pr2(0) == 1);
    CHECK(pr2(1) == 1);
    CHECK(pr2(2) == 1);
    CHECK(pr2(3) == 10);

    A(2, 1) += 5;
    // slice of rows
    CHECK(pr(0) == 11);
    CHECK(pr(1) == 1);
    CHECK(pr(2) == 1 + 5);

    // slice of cols
    CHECK(pr2(0) == 1);
    CHECK(pr2(1) == 1 + 5);
    CHECK(pr2(2) == 1);
    CHECK(pr2(3) == 10);
  }
  {
    Matrix<double, 3, 3> A =
        expression::nullary::uniform_random_view<double>({});
    Matrix<double, 3, 3> B =
        expression::nullary::uniform_random_view<double>({});

    auto C = A * B;

    auto D = (as_tensor(A) * as_tensor(B)).eval();
    const auto &v = D.view();

    auto pr = expression::unary::PartialReduction<
        decltype(v), expression::reductions::Trace, 1, 2>(v);

    fmt::print("Partial reduction\n");
    print2(pr);

    ArrayBase ar(pr);
    ArrayBase c(C.view());

    // fmt::print("{}\n",ar);
    // fmt::print("{}\n",c);
    //  TODO: this could/should be equal right?
    CHECK((ar - c).norm() < 1e-15);
  }
}
