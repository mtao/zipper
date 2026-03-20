
#include <print>

#include "../../catch_include.hpp"
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/nullary/Random.hpp>
#include <zipper/expression/reductions/CoefficientSum.hpp>
#include <zipper/expression/reductions/Trace.hpp>
#include <zipper/expression/unary/PartialReduction.hpp>
#include <zipper/expression/unary/detail/PartialReductionDispatcher.hpp>
#include <zipper/utils/format.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/ArrayBase.hpp>

namespace {
template <typename T> void print2(T const &M) {
  if constexpr (T::extents_type::rank() == 1) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
      std::print("{} ", M(j));
    }
    std::println("");
  } else if constexpr (T::extents_type::rank() == 2) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
      for (zipper::index_type k = 0; k < M.extent(1); ++k) {
        std::print("{} ", M(j, k));
      }
      std::println("");
    }
  }
}

template <typename T>
void print(T const &M)
  requires(std::decay_t<T>::rank() == 2)
{
  std::println("{}", M);
  for (zipper::index_type j = 0; j < M.extent(0); ++j) {
    for (zipper::index_type k = 0; k < M.extent(1); ++k) {
      std::print("{} ", M(j, k));
    }
    std::println("");
  }
}

template <typename T>
void print(T const &M)
  requires(std::decay_t<T>::rank() == 1)
{
  std::println("{}", M);
  for (zipper::index_type j = 0; j < M.extent(0); ++j) {
    std::print("{} ", M(j));
  }
  std::println("");
}
} // namespace
  //
using namespace zipper;

namespace {
template <typename ExpressionType, int index>
auto make_sum(const ExpressionType &a, std::integral_constant<int, index>) {
  return expression::unary::detail::PartialReductionDispatcher<
             const ExpressionType, index>(a)
      .sum();
}
template <typename ExpressionType, int index>
auto make_norm(const ExpressionType &a, std::integral_constant<int, index>) {
  return expression::unary::detail::PartialReductionDispatcher<
             const ExpressionType, index>(a)
      .norm();
}
} // namespace

TEST_CASE("test_partial_sum", "[expression][unary]") {
  SECTION("identity_row_sum") {
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
  SECTION("mdarray_partial_sums") {
    expression::nullary::MDArray<double, zipper::extents<3, 4>> A;
    A.assign(expression::nullary::Identity<double, 3, 4>());

    auto pr = make_sum(A, std::integral_constant<int, 1>{});
    auto prn = make_norm(A, std::integral_constant<int, 1>{});
    print2(pr);
    print2(prn);

    REQUIRE(pr.extents().rank() == 1);
    REQUIRE(pr.static_extent(0) == 3);
    REQUIRE(pr.extent(0) == 3);
    REQUIRE(prn.extents().rank() == 1);
    REQUIRE(prn.static_extent(0) == 3);
    REQUIRE(prn.extent(0) == 3);

    auto pr2 = make_sum(A, std::integral_constant<int, 0>{});
    auto pr2n = make_norm(A, std::integral_constant<int, 0>{});

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
  SECTION("trace_and_direct_partial_reduction") {
    // Test using PartialReduction and Trace directly on MDArray
    using MDA = expression::nullary::MDArray<double, zipper::extents<3, 3>>;
    MDA A;
    A.assign(expression::nullary::Identity<double, 3, 3>());

    // Trace as a full reduction should be 3
    auto trace_val = expression::reductions::Trace<const MDA>(A)();
    CHECK(trace_val == 3.0);

    // Modify diagonal
    A(1, 1) = 7.0;
    auto trace_val2 = expression::reductions::Trace<const MDA>(A)();
    CHECK(trace_val2 == 9.0); // 1 + 7 + 1

    // Partial sum along axis 0 (column sums)
    auto col_sums = make_sum(A, std::integral_constant<int, 0>{});
    // Identity with A(1,1)=7: col0=1, col1=7, col2=1
    CHECK(col_sums(0) == 1.0);
    CHECK(col_sums(1) == 7.0);
    CHECK(col_sums(2) == 1.0);

    // Partial sum along axis 1 (row sums)
    auto row_sums = make_sum(A, std::integral_constant<int, 1>{});
    // row0: 1+0+0=1, row1: 0+7+0=7, row2: 0+0+1=1
    CHECK(row_sums(0) == 1.0);
    CHECK(row_sums(1) == 7.0);
    CHECK(row_sums(2) == 1.0);

    // Modify off-diagonal and verify live update
    A(0, 2) = 3.0;
    CHECK(row_sums(0) == 4.0); // 1+0+3=4
    CHECK(col_sums(2) == 4.0); // 3+0+1=4
  }
}

TEST_CASE("test_rowwise_colwise_matrix", "[matrix][storage][dense]") {
  Matrix<double, 3, 5> N;
  N = expression::nullary::uniform_random<double>(
      extents<3, 5>{}, -1, 1);

  {
    auto colnorm = N.colwise().norm();
    auto rownorm = N.rowwise().norm();
    for (index_type j = 0; j < N.extent(1); ++j) {
      CHECK(N.col(j).norm() == colnorm(j));
    }
    for (index_type j = 0; j < N.extent(0); ++j) {
      CHECK(N.row(j).norm() == rownorm(j));
    }
  }
}
