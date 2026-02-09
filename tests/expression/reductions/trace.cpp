

#include <iostream>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/Random.hpp>
#include <zipper/expression/unary/PartialTrace.hpp>

#include "../../catch_include.hpp"

namespace {
void print(zipper::concepts::Matrix auto const &M) {
  for (zipper::index_type j = 0; j < M.extent(0); ++j) {
    for (zipper::index_type k = 0; k < M.extent(1); ++k) {
      std::cout << M(j, k) << " ";
    }
    std::cout << std::endl;
  }
}
} // namespace

// === From test_matrix.cpp: test_trace ===

TEST_CASE("test_trace", "[matrix][storage][dense]") {
  zipper::Matrix<double, 3, 3> N;
  N = zipper::expression::nullary::uniform_random<double>(
      zipper::extents<3, 3>{}, -1, 1);
  N.diagonal() = zipper::expression::nullary::Constant<double, 3>(0.0);
  CHECK(N.trace() == 0);

  N.diagonal() = zipper::expression::nullary::Constant<double, 3>(1.0);
  print(N);
  CHECK(N.trace() == 3);
  N(0, 0) = 2;
  CHECK(N.trace() == 4);
  N(1, 1) = 2;
  CHECK(N.trace() == 5);
  N(2, 2) = 2;
  CHECK(N.trace() == 6);

  N.diagonal() = zipper::expression::nullary::Constant(1.0);
  print(N);
  CHECK(N.trace() == 3);
  N(0, 0) = 2;
  CHECK(N.trace() == 4);
  N(1, 1) = 2;
  CHECK(N.trace() == 5);
  N(2, 2) = 2;
  CHECK(N.trace() == 6);
}

// === From test_matrix.cpp: test_partial_trace_matrix ===

TEST_CASE("test_partial_trace_matrix", "[matrix][storage][dense]") {
  zipper::Matrix<double, 3, 3> N;
  N = zipper::expression::nullary::uniform_random<double>(
      zipper::extents<3, 3>{}, -1, 1);
  // fmt::print("Random matrix n:\n");
  print(N);

  N.diagonal() = zipper::expression::nullary::Constant<double, 3>(0.0);
  CHECK(N.trace() == 0);

  N.diagonal() = zipper::expression::nullary::Constant<double, 3>(1.0);

  {
    zipper::MatrixBase empty_partial_trace =
        zipper::expression::unary::PartialTrace<
            std::decay_t<decltype(N.expression())>>(N.expression());
    static_assert(
        std::decay_t<decltype(empty_partial_trace.extents())>::rank() == 2);
    using reducer = std::decay_t<
        decltype(empty_partial_trace.expression())>::traits::index_remover;
    constexpr static auto f2r = reducer::full_rank_to_reduced_indices;
    constexpr static auto r2f = reducer::reduced_rank_to_full_indices;
    static_assert(f2r.size() == 2);
    static_assert(r2f.size() == 2);
    static_assert(f2r[0] == 0);
    static_assert(f2r[1] == 1);
    static_assert(r2f[0] == 0);
    static_assert(f2r[1] == 1);
    static_assert(
        std::decay_t<decltype(empty_partial_trace.extents())>::static_extent(
            0) == 3);
    static_assert(
        std::decay_t<decltype(empty_partial_trace.extents())>::static_extent(
            1) == 3);
    CHECK(empty_partial_trace == N);
  }

  print(N);
  CHECK(N.trace() == 3);
  N(0, 0) = 2;
  CHECK(N.trace() == 4);
  N(1, 1) = 2;
  CHECK(N.trace() == 5);
  N(2, 2) = 2;
  CHECK(N.trace() == 6);
}
