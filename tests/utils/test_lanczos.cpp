
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/detail/dot.hpp>
#include <zipper/utils/krylov/lanczos.hpp>

#include "../catch_include.hpp"

using namespace zipper;

TEST_CASE("lanczos static 3x3 symmetric", "[krylov][lanczos]") {
    // Symmetric tridiagonal: eigenvalues are distinct → full 3D Krylov subspace.
    Matrix<double, 3, 3> M{
        {2.0, 1.0, 0.0},
        {1.0, 3.0, 1.0},
        {0.0, 1.0, 4.0},
    };
    Vector<double, 3> v1{1.0, 0.0, 0.0};

    auto result = utils::krylov::lanczos(M, v1, index_type{3});

    // V columns should be orthonormal
    for (index_type i = 0; i < 3; ++i) {
        CHECK(Vector<double, 3>(result.V.col(i)).norm() ==
              Catch::Approx(1.0).epsilon(1e-10));
        for (index_type j = 0; j < i; ++j) {
            CHECK(utils::detail::dot(
                      Vector<double, 3>(result.V.col(i)),
                      Vector<double, 3>(result.V.col(j))) ==
                  Catch::Approx(0.0).margin(1e-10));
        }
    }
}

TEST_CASE("lanczos T_mat is symmetric tridiagonal", "[krylov][lanczos]") {
    Matrix<double, 3, 3> M{
        {2.0, 1.0, 0.0},
        {1.0, 3.0, 1.0},
        {0.0, 1.0, 2.0},
    };
    Vector<double, 3> v1{1.0, 0.0, 0.0};

    auto result = utils::krylov::lanczos(M, v1, index_type{3});

    // T should be symmetric
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(result.T_mat(i, j) == Catch::Approx(result.T_mat(j, i)).margin(1e-12));
        }
    }

    // T should be tridiagonal: T(i,j) = 0 for |i-j| > 1
    // For a 3x3 matrix, only T(0,2) and T(2,0) should be zero
    CHECK(result.T_mat(0, 2) == Catch::Approx(0.0).margin(1e-12));
    CHECK(result.T_mat(2, 0) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("lanczos Hessenberg relation V^T M V = T", "[krylov][lanczos]") {
    Matrix<double, 3, 3> M{
        {4.0, 1.0, 1.0},
        {1.0, 4.0, 1.0},
        {1.0, 1.0, 4.0},
    };
    Vector<double, 3> v1{1.0, 0.0, 0.0};

    auto result = utils::krylov::lanczos(M, v1, index_type{3});

    // V^T M V should equal T
    auto VtMV = Matrix<double, 3, 3>(
        Matrix<double, 3, 3>(result.V.transpose() * M) * result.V);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(VtMV(i, j) == Catch::Approx(result.T_mat(i, j)).margin(1e-10));
        }
    }
}

TEST_CASE("lanczos diagonal matrix early termination", "[krylov][lanczos]") {
    // Diagonal matrix: Krylov subspace from e_0 is span{e_0},
    // so after one step beta = 0 and we terminate early.
    Matrix<double, 3, 3> M{
        {5.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    Vector<double, 3> v1{1.0, 0.0, 0.0};

    auto result = utils::krylov::lanczos(M, v1, index_type{3});

    // T(0,0) should be the eigenvalue 5
    CHECK(result.T_mat(0, 0) == Catch::Approx(5.0).epsilon(1e-12));
    // beta_1 should be zero (early termination)
    CHECK(result.T_mat(1, 0) == Catch::Approx(0.0).margin(1e-12));
    CHECK(result.T_mat(0, 1) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("lanczos static N", "[krylov][lanczos]") {
    Matrix<double, 3, 3> M{
        {2.0, 1.0, 0.0},
        {1.0, 3.0, 1.0},
        {0.0, 1.0, 2.0},
    };
    Vector<double, 3> v1{1.0, 0.0, 0.0};

    auto result = utils::krylov::lanczos(M, v1, static_index_t<3>{});

    // Should produce the same result as dynamic N
    for (index_type i = 0; i < 3; ++i) {
        CHECK(Vector<double, 3>(result.V.col(i)).norm() ==
              Catch::Approx(1.0).epsilon(1e-10));
    }
}

TEST_CASE("lanczos 2x2 eigenvalues from T", "[krylov][lanczos]") {
    // [[2, 1], [1, 2]]: eigenvalues 3, 1
    Matrix<double, 2, 2> M{{2.0, 1.0}, {1.0, 2.0}};
    Vector<double, 2> v1{1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0)};

    auto result = utils::krylov::lanczos(M, v1, index_type{2});

    // The eigenvalues of T should approximate those of M.
    // For this starting vector, the first Rayleigh quotient T(0,0) = v1^T M v1 = 3
    CHECK(result.T_mat(0, 0) == Catch::Approx(3.0).epsilon(1e-10));
}

TEST_CASE("lanczos dynamic matrix", "[krylov][lanczos]") {
    MatrixXX<double> M(3, 3);
    M(0, 0) = 4.0; M(0, 1) = 1.0; M(0, 2) = 0.0;
    M(1, 0) = 1.0; M(1, 1) = 4.0; M(1, 2) = 1.0;
    M(2, 0) = 0.0; M(2, 1) = 1.0; M(2, 2) = 4.0;

    VectorX<double> v1(3);
    v1(0) = 1.0; v1(1) = 0.0; v1(2) = 0.0;

    auto result = utils::krylov::lanczos(M, v1, index_type{3});

    // V^T M V ≈ T
    auto VtMV = MatrixXX<double>(
        MatrixXX<double>(result.V.transpose() * M) * result.V);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(VtMV(i, j) == Catch::Approx(result.T_mat(i, j)).margin(1e-10));
        }
    }
}
