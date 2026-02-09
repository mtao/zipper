

#include <iostream>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/Random.hpp>
#include <zipper/expression/unary/Swizzle.hpp>

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
void print(zipper::concepts::Vector auto const &M) {
  for (zipper::index_type j = 0; j < M.extent(0); ++j) {
    std::cout << M(j) << " ";
  }
  std::cout << std::endl;
}
} // namespace

// === From expression/test_swizzle.cpp ===

TEST_CASE("test_swizzle_pair", "[swizzle]") {
    using Swizzler = zipper::detail::extents::ExtentsSwizzler<1, 0>;

    REQUIRE(Swizzler::size == 2);
    REQUIRE(Swizzler::valid_indices_rank == 2);
    REQUIRE(Swizzler::valid_internal_indices.size() == 2);
    CHECK(Swizzler::valid_internal_indices[0] == 1);
    CHECK(Swizzler::valid_internal_indices[1] == 0);

    auto a = Swizzler::unswizzle(4, 5);
    CHECK(Swizzler::_unswizzle_single_index<0>(std::make_tuple(4, 5)) == 5);
    CHECK(Swizzler::_unswizzle_single_index<1>(std::make_tuple(4, 5)) == 4);
        REQUIRE(std::tuple_size_v<decltype(a)> == 2);
        CHECK(std::get<0>(a) == 5);
        CHECK(std::get<1>(a) == 4);
}
TEST_CASE("test_swizzle_lift", "[swizzle]") {
    {
        using Swizzler =
            zipper::detail::extents::ExtentsSwizzler<std::dynamic_extent, 0>;

        REQUIRE(Swizzler::size == 2);
        REQUIRE(Swizzler::valid_indices_rank == 1);
        REQUIRE(Swizzler::valid_internal_indices.size() == 1);
        CHECK(Swizzler::valid_internal_indices[0] == 1);

        CHECK(Swizzler::swizzle_extents_type::static_extent(
                  Swizzler::valid_internal_indices[0]) == 0);

        CHECK(Swizzler::_unswizzle_single_index<0>(std::make_tuple(4, 5)) == 5);
        auto a = Swizzler::unswizzle(4, 5);
        REQUIRE(std::tuple_size_v<decltype(a)> == 1);
        CHECK(std::get<0>(a) == 5);
    }
    {
        using Swizzler =
            zipper::detail::extents::ExtentsSwizzler<0, std::dynamic_extent>;

        REQUIRE(Swizzler::size == 2);
        REQUIRE(Swizzler::valid_indices_rank == 1);
        REQUIRE(Swizzler::valid_internal_indices.size() == 1);
        CHECK(Swizzler::valid_internal_indices[0] == 0);
        CHECK(Swizzler::swizzle_extents_type::static_extent(
                  Swizzler::valid_internal_indices[0]) == 0);
        CHECK(Swizzler::_unswizzle_single_index<0>(std::make_tuple(4, 5)) == 4);

        auto a = Swizzler::unswizzle(4, 5);
        REQUIRE(std::tuple_size_v<decltype(a)> == 1);
        CHECK(std::get<0>(a) == 4);
    }

    {
        using Swizzler =
            zipper::detail::extents::ExtentsSwizzler<1, std::dynamic_extent, 0>;

        REQUIRE(Swizzler::size == 3);
        REQUIRE(Swizzler::valid_indices_rank == 2);
        REQUIRE(Swizzler::valid_internal_indices.size() == 2);
        CHECK(Swizzler::valid_internal_indices[0] == 2);
        CHECK(Swizzler::valid_internal_indices[1] == 0);

        auto a = Swizzler::unswizzle(4, 5, 6);
        CHECK(Swizzler::_unswizzle_single_index<0>(std::make_tuple(4, 5, 6)) ==
              6);
        CHECK(Swizzler::_unswizzle_single_index<1>(std::make_tuple(4, 5, 6)) ==
              4);
        REQUIRE(std::tuple_size_v<decltype(a)> == 2);
        CHECK(std::get<0>(a) == 6);
        CHECK(std::get<1>(a) == 4);
    }
}

// === From test_matrix.cpp: test_identity2 (vector swizzle tests) ===

TEST_CASE("test_identity2", "[matrix][vector][lift]") {
  zipper::Vector<double, 3> x;
  x(0) = 2;
  x(1) = 5;
  x(2) = 9;
  auto colMat = x.swizzle<zipper::MatrixBase, 0, std::dynamic_extent>();
  auto rowMat = x.swizzle<zipper::MatrixBase, std::dynamic_extent, 0>();
  // fmt::print("Vector x is: {}\n");
  REQUIRE(x.extents().rank() == 1);
  REQUIRE(x.extent(0) == 3);
  CHECK(x(0) == 2);
  CHECK(x(1) == 5);
  CHECK(x(2) == 9);
  print(x);
  // fmt::print("Vector x as colmat:\n");
  REQUIRE(colMat.extents().rank() == 2);
  REQUIRE(colMat.extent(0) == 3);
  REQUIRE(colMat.extent(1) == 1);

  CHECK(colMat(0, 0) == 2);
  CHECK(colMat(1, 0) == 5);
  CHECK(colMat(2, 0) == 9);
  print(colMat);
  // fmt::print("Vector x as rowmat:\n");
  REQUIRE(rowMat.extents().rank() == 2);
  REQUIRE(rowMat.extent(0) == 1);
  REQUIRE(rowMat.extent(1) == 3);
  CHECK(rowMat(0, 0) == 2);
  CHECK(rowMat(0, 1) == 5);
  CHECK(rowMat(0, 2) == 9);
  print(rowMat);
}

// === From test_matrix.cpp: test_transpose ===

TEST_CASE("test_transpose", "[matrix][storage][dense]") {
  zipper::Matrix<double, 3, std::dynamic_extent> M(3);
  zipper::Matrix<double, 3, 5> N;

  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 3; ++k) {
      M(j, k) = 3 * j + k;
    }
  }

  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 5; ++k) {
      N(j, k) = 5 * j + k;
    }
  }

  REQUIRE(M.transpose().extent(0) == 3);
  REQUIRE(M.transpose().extent(1) == 3);
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 3; ++k) {
      CHECK(M.transpose()(k, j) == 3 * j + k);
    }
  }

  REQUIRE(N.transpose().extent(0) == 5);
  REQUIRE(N.transpose().extent(1) == 3);

  REQUIRE(N.transpose().extents().static_extent(0) == 5);
  REQUIRE(N.transpose().extents().static_extent(1) == 3);
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 5; ++k) {
      CHECK(N.transpose()(k, j) == 5 * j + k);
    }
  }
  zipper::Matrix MTM = M.transpose() * M;
  REQUIRE(MTM.transpose().extent(0) == 3);
  REQUIRE(MTM.transpose().extent(1) == 3);

  REQUIRE(MTM.transpose().extents().static_extent(0) == std::dynamic_extent);
  REQUIRE(MTM.transpose().extents().static_extent(1) == std::dynamic_extent);
  print(MTM);
  CHECK(MTM(0, 0) == 45);
  CHECK(MTM(1, 0) == 54);
  CHECK(MTM(2, 0) == 63);
  CHECK(MTM(0, 1) == 54);
  CHECK(MTM(1, 1) == 66);
  CHECK(MTM(2, 1) == 78);
  CHECK(MTM(0, 2) == 63);
  CHECK(MTM(1, 2) == 78);
  CHECK(MTM(2, 2) == 93);

  zipper::Matrix MMT = M * M.transpose();
  REQUIRE(MMT.transpose().extent(0) == 3);
  REQUIRE(MMT.transpose().extent(1) == 3);

  REQUIRE(MMT.transpose().extents().static_extent(0) == 3);
  REQUIRE(MMT.transpose().extents().static_extent(1) == 3);
  print(MMT);
  CHECK(MMT(0, 0) == 5);
  CHECK(MMT(1, 0) == 14);
  CHECK(MMT(2, 0) == 23);
  CHECK(MMT(0, 1) == 14);
  CHECK(MMT(1, 1) == 50);
  CHECK(MMT(2, 1) == 86);
  CHECK(MMT(0, 2) == 23);
  CHECK(MMT(1, 2) == 86);
  CHECK(MMT(2, 2) == 149);
}

// === From test_matrix.cpp: test_matrix_transpose ===

TEST_CASE("test_matrix_transpose", "[matrix][storage][dense]") {
  {
    zipper::Matrix<double, 4, 4> I =
        zipper::expression::nullary::uniform_random<double>({});

    auto I2 = I.transpose();
    auto I3 = I.transpose().eval();

    CHECK(I2 == I3);
    for (zipper::index_type j = 0; j < 4; ++j) {
      for (zipper::index_type k = 0; k < 4; ++k) {
        CHECK(I(j, k) == I2(k, j));
      }
    }

    zipper::Vector<double, 4> V =
        zipper::expression::nullary::uniform_random<double>({});
    auto R1 = (I2 * V);
    auto R2 = (I2 * V).eval();
    auto R3 = (I3 * V).eval();
    CHECK(R1 == R3);
    CHECK(R2 == R3);
  }
  {
    zipper::Matrix<double, std::dynamic_extent, std::dynamic_extent> I =
        zipper::expression::nullary::uniform_random<double>(
            zipper::create_dextents(3, 3));

    auto I2 = I.transpose();
    auto I3 = I.transpose().eval();
    CHECK(I2 == I3);
    for (zipper::index_type j = 0; j < I.extent(0); ++j) {
      for (zipper::index_type k = 0; k < I.extent(1); ++k) {
        CHECK(I(j, k) == I2(k, j));
      }
    }
    zipper::Vector<double, 3> V =
        zipper::expression::nullary::uniform_random<double>({});
    auto R1 = (I2 * V);
    auto R2 = (I2 * V).eval();
    auto R3 = (I3 * V).eval();
    CHECK(R1 == R3);
    CHECK(R2 == R3);
  }

  {
    zipper::Matrix<double, 4, 4> I =
        zipper::expression::nullary::uniform_random<double>({});
    const auto S =
        I.slice(zipper::static_slice<0, 3>(), zipper::static_slice<0, 3>());

    auto I2 = S.transpose();
    auto I3 = S.transpose().eval();

    CHECK(I2 == I3);
    for (zipper::index_type j = 0; j < S.extent(0); ++j) {
      for (zipper::index_type k = 0; k < S.extent(1); ++k) {
        CHECK(S(j, k) == I2(k, j));
      }
    }
  }
  {
    const zipper::Matrix<double, 4, 4> I =
        zipper::expression::nullary::uniform_random<double>({});
    const auto S =
        I.slice(zipper::static_slice<0, 3>(), zipper::static_slice<0, 3>());

    auto I2 = S.transpose();
    auto I3 = S.transpose().eval();

    const auto &I2view = I2.expression();
    const auto &I2sliceview = I2view.expression();
    using slice_extents_type =
        typename std::decay_t<decltype(I2sliceview)>::extents_type;
    static_assert(slice_extents_type::rank() == 2);
    static_assert(slice_extents_type::static_extent(0) == 3);
    static_assert(slice_extents_type::static_extent(1) == 3);

    using swizzle_extents_type =
        typename std::decay_t<decltype(I2view)>::extents_type;
    static_assert(swizzle_extents_type::rank() == 2);
    static_assert(swizzle_extents_type::static_extent(0) == 3);
    static_assert(swizzle_extents_type::static_extent(1) == 3);

    using swizzler_type =
        typename std::decay_t<decltype(I2view)>::swizzler_type;
    auto s = swizzler_type::unswizzle(1, 0);
    CHECK(std::get<0>(s) == 0);
    CHECK(std::get<1>(s) == 1);
    CHECK(I2 == I3);
    for (zipper::index_type j = 0; j < S.extent(0); ++j) {
      for (zipper::index_type k = 0; k < S.extent(1); ++k) {
        CHECK(S(j, k) == I2(k, j));
      }
    }
  }
  {
    zipper::Matrix<double, std::dynamic_extent, std::dynamic_extent> I =
        zipper::expression::nullary::uniform_random<double>(
            zipper::create_dextents(5, 5));
    const auto S =
        I.slice(zipper::static_slice<0, 3>(), zipper::static_slice<0, 3>());

    auto I2 = S.transpose();
    auto I3 = S.transpose().eval();
    print(I);
    print(I2);
    print(I3);
    CHECK(I2 == I3);
    for (zipper::index_type j = 0; j < S.extent(0); ++j) {
      for (zipper::index_type k = 0; k < S.extent(1); ++k) {
        CHECK(S(j, k) == I2(k, j));
      }
    }
  }
}
