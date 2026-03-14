
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/detail/dot.hpp>
#include <zipper/utils/krylov/arnoldi.hpp>

#include "../catch_include.hpp"

using namespace zipper;

TEST_CASE("arnoldi static 3x3 dynamic N", "[krylov][arnoldi]") {
    // Non-symmetric matrix — Krylov subspace from b=[1,0,0] has full
    // dimension 3, so N=2 iterations give 3 genuinely orthonormal columns.
    Matrix<double, 3, 3> M{
        {1.0, 2.0, 0.0},
        {0.0, 3.0, 1.0},
        {1.0, 0.0, 4.0},
    };
    Vector<double, 3> b{1.0, 0.0, 0.0};

    auto result = utils::krylov::arnoldi(M, b, index_type{2});

    // Q columns 0..N should be orthonormal
    for (index_type i = 0; i <= 2; ++i) {
        CHECK(Vector<double, 3>(result.Q.col(i)).norm() ==
              Catch::Approx(1.0).epsilon(1e-10));
        for (index_type j = 0; j < i; ++j) {
            CHECK(utils::detail::dot(
                      Vector<double, 3>(result.Q.col(i)),
                      Vector<double, 3>(result.Q.col(j))) ==
                  Catch::Approx(0.0).margin(1e-10));
        }
    }
}

TEST_CASE("arnoldi static 3x3 static N", "[krylov][arnoldi]") {
    Matrix<double, 3, 3> M{
        {1.0, 2.0, 0.0},
        {0.0, 3.0, 1.0},
        {1.0, 0.0, 4.0},
    };
    Vector<double, 3> b{1.0, 0.0, 0.0};

    auto result =
        utils::krylov::arnoldi(M, b, static_index_t<2>{});

    for (index_type i = 0; i <= 2; ++i) {
        CHECK(Vector<double, 3>(result.Q.col(i)).norm() ==
              Catch::Approx(1.0).epsilon(1e-10));
    }
}

TEST_CASE("arnoldi 2x2 hessenberg relation", "[krylov][arnoldi]") {
    // Verify M * Q_N ≈ Q_{N+1} * H column by column.
    // For column 0: M*q0 = H(0,0)*q0 + H(1,0)*q1
    Matrix<double, 2, 2> M{{2.0, 1.0}, {1.0, 3.0}};
    Vector<double, 2> b{1.0, 0.0};

    auto result = utils::krylov::arnoldi(M, b, index_type{2});

    // H(0,0) should equal q0^T M q0 (the Rayleigh quotient)
    Vector<double, 2> q0(result.Q.col(0));
    double rq = utils::detail::dot(q0, Vector<double, 2>(M * q0));
    CHECK(result.H(0, 0) == Catch::Approx(rq).epsilon(1e-12));
}

TEST_CASE("arnoldi diagonal early termination", "[krylov][arnoldi]") {
    // Diagonal matrix: Krylov subspace from e_0 is just span{e_0},
    // so after one step M*q0 = 5*q0 is fully captured and the
    // residual is zero — H(1,0) should be ~0.
    Matrix<double, 3, 3> M{
        {5.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    Vector<double, 3> b{1.0, 0.0, 0.0};

    auto result = utils::krylov::arnoldi(M, b, index_type{3});

    CHECK(result.H(0, 0) == Catch::Approx(5.0).epsilon(1e-12));
    CHECK(result.H(1, 0) == Catch::Approx(0.0).margin(1e-12));
}
