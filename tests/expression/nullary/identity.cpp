
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/Random.hpp>
#include <zipper/Matrix.hpp>

#include "../../catch_include.hpp"

using namespace zipper::expression::nullary;
using namespace zipper;

TEST_CASE("test_identity", "[expression][nullary]") {

  Identity<double, std::dynamic_extent> a(create_dextents(3));
  Identity<double, 3, 3> b;
  Identity<double> c;

  REQUIRE(a.extents().rank() == 1);
  CHECK(a.extent(0) == 3);
  REQUIRE(b.extents().rank() == 2);
  CHECK(b.extent(0) == 3);
  CHECK(b.extent(1) == 3);

  REQUIRE(c.extents().rank() == 0);

  CHECK(a(0) == 1);
  CHECK(a(1) == 0);
  CHECK(a(2) == 0);

  CHECK(b(0, 0) == 1);
  CHECK(b(0, 1) == 0);
  CHECK(b(0, 2) == 0);
  CHECK(b(1, 0) == 0);
  CHECK(b(1, 1) == 1);
  CHECK(b(1, 2) == 0);

  // for an "infinite" view we can pass in random values
  CHECK(c(0) == 1);
  CHECK(c(2) == 0);
  CHECK(c(1, 2) == 0);
  CHECK(c(2, 2) == 1);
  CHECK(c(2, 3, 1, 2) == 0);
  CHECK(c(3, 3, 3, 3) == 1);
  CHECK(c() == 1);
}

// === From test_matrix.cpp: test_identity (Identity * Matrix properties) ===

TEST_CASE("test_identity_matrix", "[matrix][identity]") {
  auto check_identity = [](const auto &i) {
    for (index_type j = 0; j < i.extent(0); ++j) {
      for (index_type k = 0; k < i.extent(1); ++k) {
        if (j == k) {
          CHECK(i(j, k) == 1);
        } else {
          CHECK(i(j, k) == 0);
        }
      }
    }
  };
  Matrix<double, 3, 3> I =
      MatrixBase(Identity<double, 3, 3>{});

  check_identity(I);
  check_identity(
      MatrixBase(Identity<double, 3, 3>{}));
  CHECK(
      (I ==
       MatrixBase(
           Identity<double, std::dynamic_extent, 3>{
               3})));
  CHECK((I ==
         MatrixBase(
             Identity<double, std::dynamic_extent,
                                                  std::dynamic_extent>{3, 3})));

  check_identity(MatrixBase(
      Identity<double, std::dynamic_extent,
                                           std::dynamic_extent>{20, 20}));

  Matrix<double, 3, 3> M =
      uniform_random<double>(
          extents<3, 3>{}, 0, 5);

  Matrix<double, 3, 3> MI = I * M;
  Matrix<double, 3, 3> IM = M * I;
  CHECK((M == MI));
  CHECK((M == IM));

  // These last unit tests check that the identity matrix view works, but it
  // was mostly written so i could add print statements in the matrix product
  // to check that sparse operations are happening properly

  MatrixBase IV = Identity<double, 3, 3>{};
  Matrix<double, 3, 3> MIV = IV * M;
  Matrix<double, 3, 3> IVM = M * IV;
  CHECK((M == MIV));
  CHECK((M == IVM));
  Matrix<double, 3, 3> IVIV = IV * IV;
  CHECK((I == IVIV));
}
