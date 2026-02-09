
#include <iostream>

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

using namespace zipper;

namespace {
template <typename Type, int index>
auto make_sum(const Type &a, std::integral_constant<int, index>) {
  return expression::unary::detail::PartialReductionDispatcher<const Type,
                                                               index>(a)
      .sum();
}
template <typename Type, int index>
auto make_norm(const Type &a, std::integral_constant<int, index>) {
  return expression::unary::detail::PartialReductionDispatcher<const Type,
                                                               index>(a)
      .norm();
}
} // namespace

TEST_CASE("test_partial_sum_identity", "[expression][unary][partial_reduction]") {
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

TEST_CASE("test_partial_sum_mdarray", "[expression][unary][partial_reduction]") {
  // Use MDArray as a Matrix replacement: 3x4 identity-like
  expression::nullary::MDArray<double, extents<3, 4>,
                               storage::layout_left,
                               default_accessor_policy<double>>
      A;
  // Initialize to identity (as much as a 3x4 can be)
  for (index_type j = 0; j < 3; ++j) {
    for (index_type k = 0; k < 4; ++k) {
      A(j, k) = (j == k) ? 1.0 : 0.0;
    }
  }

  // Sum along axis 1 (sum of each row)
  auto pr = make_sum(A, std::integral_constant<int, 1>{});
  auto prn = make_norm(A, std::integral_constant<int, 1>{});

  REQUIRE(pr.extents().rank() == 1);
  REQUIRE(pr.static_extent(0) == 3);
  REQUIRE(pr.extent(0) == 3);
  REQUIRE(prn.extents().rank() == 1);
  REQUIRE(prn.static_extent(0) == 3);
  REQUIRE(prn.extent(0) == 3);

  // Sum along axis 0 (sum of each column)
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

  // update a value — partial reduction is a lazy expression over A
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

// TODO: The following test depends on Matrix, ArrayBase, as_tensor, and operator*
// which are still in the unmigrated views:: namespace. Re-enable once those are migrated.
//
// TEST_CASE("test_partial_reduction_trace", "[expression][unary]") {
//   Matrix<double, 3, 3> A = ...;
//   Matrix<double, 3, 3> B = ...;
//   auto C = A * B;
//   auto D = (as_tensor(A) * as_tensor(B)).eval();
//   const auto &v = D.view();
//   auto pr = expression::unary::PartialReduction<
//       decltype(v), expression::reductions::Trace, 1, 2>(v);
//   ...
// }

// === From test_matrix.cpp: test_rowwise_colwise_matrix ===

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
