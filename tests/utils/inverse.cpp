

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

    auto I2_r = zipper::utils::inverse(I);

    REQUIRE(I2_r);

    auto I2 = *I2_r;
    CHECK(I == I2);
  }
  {
    zipper::Matrix<double, 2, 2> I =
        zipper::expression::nullary::Identity<double, 2, 2>{};

    auto I2_r = zipper::utils::inverse(I);

    REQUIRE(I2_r);

    auto I2 = *I2_r;
    CHECK(I == I2);
  }
  {
    zipper::Matrix<double, std::dynamic_extent, std::dynamic_extent> I =
        zipper::expression::nullary::Identity<double, 2, 2>{};

    auto I2_r = zipper::utils::inverse(I);

    REQUIRE(I2_r);

    auto I2 = *I2_r;
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
      auto Inv_r = zipper::utils::inverse(M);
      REQUIRE(Inv_r);
      auto Inv = *Inv_r;
      auto Inv2_r = zipper::utils::inverse(M2);
      REQUIRE(Inv2_r);
      auto Inv2 = *Inv2_r;

      auto myI = M * Inv;
      CHECK(as_array((I - myI)).norm() < 1e-5);

      CHECK(Inv(a, a) == Inv2(0, 0));
      CHECK(Inv(a, b) == Inv2(0, 1));
      CHECK(Inv(b, a) == Inv2(1, 0));
      CHECK(Inv(b, b) == Inv2(1, 1));
    }
  }
  {
    zipper::Matrix<double, 2, 2> M{{4.0, 1.0}, {2.0, 3.0}};
    auto Inv_r = zipper::utils::inverse(M);
    REQUIRE(Inv_r);
    auto Inv = *Inv_r;

    zipper::MatrixBase I = zipper::expression::nullary::Identity<double, 2, 2>();

    auto myI = M * Inv;
    auto myI2 = Inv * M;

    CHECK(as_array((I - myI)).norm() < 1e-5);
    CHECK(as_array((I - myI2)).norm() < 1e-5);
  }
  {
    zipper::Matrix<double, 3, 3> M{
        {4.0, 1.0, 0.0}, {1.0, 4.0, 1.0}, {0.0, 1.0, 4.0}};
    auto Inv_r = zipper::utils::inverse(M);
    REQUIRE(Inv_r);
    auto Inv = *Inv_r;

    zipper::MatrixBase I = zipper::expression::nullary::Identity<double, 3, 3>();

    auto myI = M * Inv;
    auto myI2 = Inv * M;

    CHECK(as_array((I - myI)).norm() < 1e-5);
    CHECK(as_array((I - myI2)).norm() < 1e-5);
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

    auto Inv_r = zipper::utils::inverse(A);

    REQUIRE(Inv_r);

    auto Inv = *Inv_r;

    // Verify A * Inv ≈ I.
    zipper::MatrixBase I = zipper::expression::nullary::Identity<double, 4, 4>();
    auto prod = A * Inv;
    CHECK(as_array((I - prod)).norm() < 1e-8);

    // Verify Inv * A ≈ I.
    auto prod2 = Inv * A;
    CHECK(as_array((I - prod2)).norm() < 1e-8);
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

    auto Inv_r = zipper::utils::inverse(A);

    REQUIRE(Inv_r);

    auto Inv = *Inv_r;

    // Verify A * Inv ≈ I.
    zipper::MatrixBase I = zipper::expression::nullary::Identity<double, 5, 5>();
    auto prod = A * Inv;
    CHECK(as_array((I - prod)).norm() < 1e-8);
}

TEST_CASE("inverse 4x4 identity", "[matrix][inverse][general]") {
    zipper::Matrix<double, 4, 4> I =
        zipper::expression::nullary::Identity<double, 4, 4>{};

    auto Inv_r = zipper::utils::inverse(I);

    REQUIRE(Inv_r);

    auto Inv = *Inv_r;

    // Inverse of identity should be identity.
    for (zipper::index_type i = 0; i < 4; ++i) {
        for (zipper::index_type j = 0; j < 4; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(Inv(i, j) == Catch::Approx(expected).margin(1e-12));
        }
    }
}

TEST_CASE("inverse reports singular matrices", "[matrix][inverse][expected]") {
    using Kind = zipper::utils::solver::SolverError::Kind;
    {
        zipper::Matrix<double, 2, 2> M{{1.0, 2.0}, {2.0, 4.0}};
        auto r = zipper::utils::inverse(M);
        REQUIRE_FALSE(r);
        CHECK(r.error().kind == Kind::breakdown);
    }
    {
        zipper::Matrix<double, 3, 3> M{{1, 2, 3}, {2, 4, 6}, {0, 1, 1}};
        auto r = zipper::utils::inverse(M);
        REQUIRE_FALSE(r);
        CHECK(r.error().kind == Kind::breakdown);
    }
    {
        zipper::Matrix<double, 4, 4> M{
            {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}};
        auto r = zipper::utils::inverse(M);
        REQUIRE_FALSE(r);
        CHECK(r.error().kind == Kind::breakdown);
    }
    {
        zipper::Matrix<double, 2, 2> M{{std::nan(""), 0.0}, {0.0, 1.0}};
        auto r = zipper::utils::inverse(M);
        REQUIRE_FALSE(r);
    }
}

TEST_CASE("inverse rejects non-square matrices", "[matrix][inverse][expected]") {
    zipper::Matrix<double, zipper::dynamic_extent, zipper::dynamic_extent> M(2, 3);
    auto r = zipper::utils::inverse(M);
    REQUIRE_FALSE(r);
    CHECK(r.error().kind ==
          zipper::utils::solver::SolverError::Kind::invalid_input);
}
