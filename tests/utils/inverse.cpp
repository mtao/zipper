

#include <iostream>
#include <zipper/Matrix.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/Random.hpp>
#include <zipper/utils/inverse.hpp>

#include "../catch_include.hpp"

TEST_CASE("test_matrix_inverse", "[matrix][storage][dense]") {
  {
    zipper::Matrix<double, 3, 3> I =
        zipper::expression::nullary::Identity<double, 3, 3>{};

    auto I2 = zipper::utils::inverse(I);
    CHECK(I == I2);
  }
  {
    zipper::Matrix<double, 2, 2> I =
        zipper::expression::nullary::Identity<double, 2, 2>{};

    auto I2 = zipper::utils::inverse(I);
    CHECK(I == I2);
  }
  {
    zipper::Matrix<double, std::dynamic_extent, std::dynamic_extent> I =
        zipper::expression::nullary::Identity<double, 2, 2>{};

    auto I2 = zipper::utils::inverse(I);
    CHECK(I == I2);
  }
  {
    zipper::MatrixBase I = zipper::expression::nullary::Identity<double, 3, 3>();
    for (auto [a, b] : {std::tuple{0, 1}, std::tuple{0, 2}, std::tuple{1, 2}}) {
      zipper::Matrix<double, 3, 3> M =
          zipper::expression::nullary::Identity<double, 3, 3>{};

      zipper::Matrix<double, 2, 2> M2;

      M(a, a) = 2;
      M(a, b) = 3;
      M(b, a) = 4;
      M(b, b) = 5;

      M2(0, 0) = 2;
      M2(0, 1) = 3;
      M2(1, 0) = 4;
      M2(1, 1) = 5;
      auto Inv = zipper::utils::inverse(M);
      auto Inv2 = zipper::utils::inverse(M2);

      auto myI = M * Inv;
      CHECK((I - myI).as_array().norm() < 1e-5);

      CHECK(Inv(a, a) == Inv2(0, 0));
      CHECK(Inv(a, b) == Inv2(0, 1));
      CHECK(Inv(b, a) == Inv2(1, 0));
      CHECK(Inv(b, b) == Inv2(1, 1));
    }
  }
  for (int j = 0; j < 10; ++j) {
    zipper::Matrix<double, 2, 2> M =
        zipper::expression::nullary::uniform_random<double>({});
    auto Inv = zipper::utils::inverse(M);

    zipper::MatrixBase I = zipper::expression::nullary::Identity<double, 2, 2>();

    auto myI = M * Inv;
    auto myI2 = Inv * M;

    CHECK((I - myI).as_array().norm() < 1e-5);
    CHECK((I - myI2).as_array().norm() < 1e-5);
  }
  for (int j = 0; j < 10; ++j) {
    zipper::Matrix<double, 3, 3> M =
        zipper::expression::nullary::uniform_random<double>({});
    auto Inv = zipper::utils::inverse(M);

    zipper::MatrixBase I = zipper::expression::nullary::Identity<double, 3, 3>();

    auto myI = M * Inv;
    auto myI2 = Inv * M;

    CHECK((I - myI).as_array().norm() < 1e-5);
    CHECK((I - myI2).as_array().norm() < 1e-5);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// General n x n inverse via QR (for n > 3)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("inverse 4x4 static extents", "[matrix][inverse][general]") {
    // A well-conditioned 4x4 matrix.
    zipper::Matrix<double, 4, 4> A{
        {4.0, 1.0, 0.0, 0.0},
        {1.0, 4.0, 1.0, 0.0},
        {0.0, 1.0, 4.0, 1.0},
        {0.0, 0.0, 1.0, 4.0},
    };

    auto Inv = zipper::utils::inverse(A);

    // Verify A * Inv ≈ I.
    zipper::MatrixBase I = zipper::expression::nullary::Identity<double, 4, 4>();
    auto prod = A * Inv;
    CHECK((I - prod).as_array().norm() < 1e-8);

    // Verify Inv * A ≈ I.
    auto prod2 = Inv * A;
    CHECK((I - prod2).as_array().norm() < 1e-8);
}

TEST_CASE("inverse 5x5 static extents", "[matrix][inverse][general]") {
    // Diagonally dominant 5x5 matrix (guaranteed well-conditioned).
    zipper::Matrix<double, 5, 5> A{
        {10.0, 1.0, 0.0, 0.0, 0.0},
        {1.0, 10.0, 1.0, 0.0, 0.0},
        {0.0, 1.0, 10.0, 1.0, 0.0},
        {0.0, 0.0, 1.0, 10.0, 1.0},
        {0.0, 0.0, 0.0, 1.0, 10.0},
    };

    auto Inv = zipper::utils::inverse(A);

    // Verify A * Inv ≈ I.
    zipper::MatrixBase I = zipper::expression::nullary::Identity<double, 5, 5>();
    auto prod = A * Inv;
    CHECK((I - prod).as_array().norm() < 1e-8);
}

TEST_CASE("inverse 4x4 identity", "[matrix][inverse][general]") {
    zipper::Matrix<double, 4, 4> I =
        zipper::expression::nullary::Identity<double, 4, 4>{};

    auto Inv = zipper::utils::inverse(I);

    // Inverse of identity should be identity.
    for (zipper::index_type i = 0; i < 4; ++i) {
        for (zipper::index_type j = 0; j < 4; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(Inv(i, j) == Catch::Approx(expected).margin(1e-12));
        }
    }
}
