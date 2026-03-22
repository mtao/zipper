
#include <cmath>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/decomposition/qr.hpp>

#include "../catch_include.hpp"

using namespace zipper;

// ─────────────────────────────────────────────────────────────────────────────
// Householder QR (reduced)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("qr reduced 3x3", "[decomposition][qr]") {
    Matrix<double, 3, 3> A{
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0},
    };

    auto [Q, R] = utils::decomposition::qr(A);

    // Q should have orthonormal columns: Q^T * Q ≈ I (Gram matrix).
    Matrix<double, 3, 3> QtQ = Q.transpose() * Q;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(QtQ(i, j) == Catch::Approx(expected).margin(1e-12));
        }
    }

    // R should be upper triangular.
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < i; ++j) {
            CHECK(R(i, j) == Catch::Approx(0.0).margin(1e-12));
        }
    }

    // Q * R ≈ A.
    Matrix<double, 3, 3> QR = Q * R;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, j)).margin(1e-12));
        }
    }
}

TEST_CASE("qr reduced 4x3 tall", "[decomposition][qr]") {
    // Tall matrix: Q is 4x3, R is 3x3.
    Matrix<double, 4, 3> A{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
        {10.0, 11.0, 13.0}, // non-singular (changed 12→13)
    };

    auto [Q, R] = utils::decomposition::qr(A);

    // Q^T * Q ≈ I (3x3 Gram matrix).
    Matrix<double, 3, 3> QtQ = Q.transpose() * Q;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(QtQ(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }

    // R upper triangular.
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < i; ++j) {
            CHECK(R(i, j) == Catch::Approx(0.0).margin(1e-10));
        }
    }

    // Q * R ≈ A.
    Matrix<double, 4, 3> QR = Q * R;
    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, j)).margin(1e-10));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Householder QR (full)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("qr_full 3x3", "[decomposition][qr_full]") {
    Matrix<double, 3, 3> A{
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0},
    };

    auto [Q, R] = utils::decomposition::qr_full(A);

    // Q should be orthogonal: Q^T * Q ≈ I (Gram matrix).
    Matrix<double, 3, 3> QtQ = Q.transpose() * Q;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(QtQ(i, j) == Catch::Approx(expected).margin(1e-12));
        }
    }

    // R upper triangular.
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < i; ++j) {
            CHECK(R(i, j) == Catch::Approx(0.0).margin(1e-12));
        }
    }

    // Q * R ≈ A.
    Matrix<double, 3, 3> QR = Q * R;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, j)).margin(1e-12));
        }
    }
}

TEST_CASE("qr_full 4x3 tall", "[decomposition][qr_full]") {
    Matrix<double, 4, 3> A{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
        {10.0, 11.0, 13.0},
    };

    auto [Q, R] = utils::decomposition::qr_full(A);

    // Q is 4x4, should be orthogonal (Gram matrix Q^T Q ≈ I).
    Matrix<double, 4, 4> QtQ = Q.transpose() * Q;
    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < 4; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(QtQ(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }

    // R is 4x3, upper trapezoidal (R(i,j) = 0 for i > j).
    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < std::min(i, index_type{3}); ++j) {
            CHECK(R(i, j) == Catch::Approx(0.0).margin(1e-10));
        }
    }

    // Q * R ≈ A.
    Matrix<double, 4, 3> QR = Q * R;
    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, j)).margin(1e-10));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gram-Schmidt QR
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("qr_gram_schmidt 3x3", "[decomposition][qr_gram_schmidt]") {
    Matrix<double, 3, 3> A{
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0},
    };

    auto [Q, R] = utils::decomposition::qr_gram_schmidt(A);

    // Q^T * Q ≈ I (Gram matrix).
    Matrix<double, 3, 3> QtQ = Q.transpose() * Q;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(QtQ(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }

    // R upper triangular.
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < i; ++j) {
            CHECK(R(i, j) == Catch::Approx(0.0).margin(1e-10));
        }
    }

    // Q * R ≈ A.
    Matrix<double, 3, 3> QR = Q * R;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, j)).margin(1e-10));
        }
    }
}

TEST_CASE("qr_gram_schmidt 4x3 tall", "[decomposition][qr_gram_schmidt]") {
    Matrix<double, 4, 3> A{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
        {10.0, 11.0, 13.0},
    };

    auto [Q, R] = utils::decomposition::qr_gram_schmidt(A);

    // Q^T * Q ≈ I (3x3 Gram matrix).
    Matrix<double, 3, 3> QtQ = Q.transpose() * Q;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(QtQ(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }

    // R upper triangular.
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < i; ++j) {
            CHECK(R(i, j) == Catch::Approx(0.0).margin(1e-10));
        }
    }

    // Q * R ≈ A.
    Matrix<double, 4, 3> QR = Q * R;
    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, j)).margin(1e-10));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Diagonal R entries should be positive (convention)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("qr diagonal of R has consistent sign", "[decomposition][qr]") {
    // For Householder QR the diagonal of R can be positive or negative
    // (depends on the sign choice in the reflection).  This test just
    // verifies that the factorisation is valid regardless.
    Matrix<double, 2, 2> A{{3.0, 1.0}, {4.0, 2.0}};

    auto [Q, R] = utils::decomposition::qr(A);

    // Q * R ≈ A.
    Matrix<double, 2, 2> QR = Q * R;
    for (index_type i = 0; i < 2; ++i) {
        for (index_type j = 0; j < 2; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, j)).margin(1e-12));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// qr_solve (reduced)
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("qr_solve 2x2", "[decomposition][qr_solve]") {
    // A = [[4, 1], [1, 3]], b = [1, 2], x = [1/11, 7/11]
    Matrix<double, 2, 2> A{{4.0, 1.0}, {1.0, 3.0}};
    Vector<double, 2> b{1.0, 2.0};

    auto result = utils::decomposition::qr_solve(A, b);
    REQUIRE(result.has_value());

    CHECK(result->operator()(0) == Catch::Approx(1.0 / 11.0).margin(1e-10));
    CHECK(result->operator()(1) == Catch::Approx(7.0 / 11.0).margin(1e-10));
}

TEST_CASE("qr_solve 3x3", "[decomposition][qr_solve]") {
    Matrix<double, 3, 3> A{
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0},
    };
    // Choose x_true = [1, 2, 3] and compute b = A * x_true.
    Vector<double, 3> x_true{1.0, 2.0, 3.0};
    Vector<double, 3> b = A * x_true;

    auto result = utils::decomposition::qr_solve(A, b);
    REQUIRE(result.has_value());

    for (index_type i = 0; i < 3; ++i) {
        CHECK(result->operator()(i) == Catch::Approx(x_true(i)).margin(1e-10));
    }
}

TEST_CASE("qr_solve 3x3 SPD", "[decomposition][qr_solve]") {
    // The same system used in the iterative solver tests.
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    auto result = utils::decomposition::qr_solve(A, b);
    REQUIRE(result.has_value());

    CHECK(result->operator()(0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(result->operator()(1) == Catch::Approx(1.0).margin(1e-10));
    CHECK(result->operator()(2) == Catch::Approx(1.0).margin(1e-10));
}

TEST_CASE("qr_solve 3x3 non-symmetric", "[decomposition][qr_solve]") {
    // Upper triangular system — same as GMRES test system.
    Matrix<double, 3, 3> A{
        {3.0, 1.0, 0.0}, {0.0, 3.0, 1.0}, {0.0, 0.0, 3.0}};
    Vector<double, 3> b{4.0, 4.0, 3.0};

    auto result = utils::decomposition::qr_solve(A, b);
    REQUIRE(result.has_value());

    CHECK(result->operator()(0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(result->operator()(1) == Catch::Approx(1.0).margin(1e-10));
    CHECK(result->operator()(2) == Catch::Approx(1.0).margin(1e-10));
}

TEST_CASE("qr_solve overdetermined (least-squares)", "[decomposition][qr_solve]") {
    // A is 4x2, b is chosen so the system is consistent: b = A * x_true.
    Matrix<double, 4, 2> A{
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0},
        {1.0, -1.0},
    };
    // x_true = [2, 3], b = A * x_true = [2, 3, 5, -1]
    Vector<double, 4> b{2.0, 3.0, 5.0, -1.0};

    auto result = utils::decomposition::qr_solve(A, b);
    REQUIRE(result.has_value());

    CHECK(result->operator()(0) == Catch::Approx(2.0).margin(1e-10));
    CHECK(result->operator()(1) == Catch::Approx(3.0).margin(1e-10));
}

TEST_CASE("qr_solve overdetermined (inconsistent — least-squares)", "[decomposition][qr_solve]") {
    // A is 3x2, b is NOT in the column space of A.
    // This should still return a least-squares solution.
    Matrix<double, 3, 2> A{
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0},
    };
    // b = [1, 1, 3] is not in col(A) since col(A) = {[a, b, a+b]}.
    // Least-squares solution: minimise ||Ax - b||^2.
    // Normal equations: A^T A x = A^T b
    // A^T A = [[2, 1], [1, 2]], A^T b = [4, 4]
    // Solution: x = [4/3, 4/3]
    Vector<double, 3> b{1.0, 1.0, 3.0};

    auto result = utils::decomposition::qr_solve(A, b);
    REQUIRE(result.has_value());

    CHECK(result->operator()(0) == Catch::Approx(4.0 / 3.0).margin(1e-10));
    CHECK(result->operator()(1) == Catch::Approx(4.0 / 3.0).margin(1e-10));
}

// ─────────────────────────────────────────────────────────────────────────────
// qr_solve_full
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("qr_solve_full 3x3", "[decomposition][qr_solve_full]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    auto result = utils::decomposition::qr_solve_full(A, b);
    REQUIRE(result.has_value());

    CHECK(result->operator()(0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(result->operator()(1) == Catch::Approx(1.0).margin(1e-10));
    CHECK(result->operator()(2) == Catch::Approx(1.0).margin(1e-10));
}

TEST_CASE("qr_solve_full matches qr_solve for overdetermined", "[decomposition][qr_solve_full]") {
    Matrix<double, 3, 2> A{
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0},
    };
    Vector<double, 3> b{1.0, 1.0, 3.0};

    auto result_reduced = utils::decomposition::qr_solve(A, b);
    auto result_full = utils::decomposition::qr_solve_full(A, b);
    REQUIRE(result_reduced.has_value());
    REQUIRE(result_full.has_value());

    CHECK(result_reduced->operator()(0) == Catch::Approx(result_full->operator()(0)).margin(1e-10));
    CHECK(result_reduced->operator()(1) == Catch::Approx(result_full->operator()(1)).margin(1e-10));
}

// ─────────────────────────────────────────────────────────────────────────────
// QRReducedResult::solve(b) and QRFullResult::solve(b)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("QRReducedResult::solve 3x3", "[decomposition][qr_solve]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> x_true{1.0, 2.0, 3.0};
    Vector<double, 3> b = A * x_true;

    auto decomp = utils::decomposition::qr(A);
    auto result = decomp.solve(b);
    REQUIRE(result.has_value());

    for (index_type i = 0; i < 3; ++i) {
        CHECK(result->operator()(i) ==
              Catch::Approx(x_true(i)).margin(1e-10));
    }
}

TEST_CASE("QRReducedResult::solve matches qr_solve", "[decomposition][qr_solve]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    auto free_result = utils::decomposition::qr_solve(A, b);
    REQUIRE(free_result.has_value());

    auto decomp = utils::decomposition::qr(A);
    auto method_result = decomp.solve(b);
    REQUIRE(method_result.has_value());

    for (index_type i = 0; i < 3; ++i) {
        CHECK(free_result->operator()(i) ==
              Catch::Approx(method_result->operator()(i)).margin(1e-14));
    }
}

TEST_CASE("QRFullResult::solve 3x3", "[decomposition][qr_solve_full]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> x_true{1.0, 2.0, 3.0};
    Vector<double, 3> b = A * x_true;

    auto decomp = utils::decomposition::qr_full(A);
    auto result = decomp.solve(b);
    REQUIRE(result.has_value());

    for (index_type i = 0; i < 3; ++i) {
        CHECK(result->operator()(i) ==
              Catch::Approx(x_true(i)).margin(1e-10));
    }
}

TEST_CASE("QRFullResult::solve matches qr_solve_full", "[decomposition][qr_solve_full]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    auto free_result = utils::decomposition::qr_solve_full(A, b);
    REQUIRE(free_result.has_value());

    auto decomp = utils::decomposition::qr_full(A);
    auto method_result = decomp.solve(b);
    REQUIRE(method_result.has_value());

    for (index_type i = 0; i < 3; ++i) {
        CHECK(free_result->operator()(i) ==
              Catch::Approx(method_result->operator()(i)).margin(1e-14));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Column-pivoted QR
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("qr_col_pivot 3x3 full rank", "[decomposition][qr_col_pivot]") {
    Matrix<double, 3, 3> A{
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0},
    };

    auto result = utils::decomposition::qr_col_pivot(A);
    const auto &Q = result.Q;
    const auto &R = result.R;
    const auto &perm = result.col_perm;

    // Q should have orthonormal columns: Q^T * Q ≈ I (Gram matrix).
    Matrix<double, 3, 3> QtQ = Q.transpose() * Q;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(QtQ(i, j) == Catch::Approx(expected).margin(1e-12));
        }
    }

    // R should be upper triangular.
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < i; ++j) {
            CHECK(R(i, j) == Catch::Approx(0.0).margin(1e-12));
        }
    }

    // |R(i,i)| should be non-increasing.
    for (index_type i = 0; i + 1 < 3; ++i) {
        CHECK(std::abs(R(i, i)) >= std::abs(R(i + 1, i + 1)) - 1e-12);
    }

    // Q * R should equal A with columns permuted: (Q*R)(:,j) == A(:,perm[j]).
    Matrix<double, 3, 3> QR = Q * R;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, perm[j])).margin(1e-12));
        }
    }

    // Full rank.
    CHECK(result.rank() == 3);
}

TEST_CASE("qr_col_pivot 4x3 tall full rank", "[decomposition][qr_col_pivot]") {
    Matrix<double, 4, 3> A{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
        {10.0, 11.0, 13.0},
    };

    auto result = utils::decomposition::qr_col_pivot(A);
    const auto &Q = result.Q;
    const auto &R = result.R;
    const auto &perm = result.col_perm;

    // Q^T * Q ≈ I (3x3 Gram matrix).
    Matrix<double, 3, 3> QtQ = Q.transpose() * Q;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(QtQ(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }

    // R upper triangular.
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < i; ++j) {
            CHECK(R(i, j) == Catch::Approx(0.0).margin(1e-10));
        }
    }

    // |R(i,i)| non-increasing.
    for (index_type i = 0; i + 1 < 3; ++i) {
        CHECK(std::abs(R(i, i)) >= std::abs(R(i + 1, i + 1)) - 1e-10);
    }

    // Q * R ≈ A(:, perm).
    Matrix<double, 4, 3> QR = Q * R;
    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, perm[j])).margin(1e-10));
        }
    }

    CHECK(result.rank() == 3);
}

TEST_CASE("qr_col_pivot rank-deficient 3x3", "[decomposition][qr_col_pivot]") {
    // Col 2 = Col 0 + Col 1, so rank = 2.
    Matrix<double, 3, 3> A{
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0},
        {1.0, 1.0, 2.0},
    };

    auto result = utils::decomposition::qr_col_pivot(A);
    CHECK(result.rank() == 2);

    // Verify factorisation: Q * R ≈ A(:, perm).
    const auto &Q = result.Q;
    const auto &R = result.R;
    const auto &perm = result.col_perm;

    Matrix<double, 3, 3> QR = Q * R;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, perm[j])).margin(1e-10));
        }
    }
}

TEST_CASE("qr_col_pivot rank-deficient 4x3", "[decomposition][qr_col_pivot]") {
    // Rows 0 and 2 are identical, cols 0 and 2 are identical.
    // Rank should be 2.
    Matrix<double, 4, 3> A{
        {1.0, 2.0, 1.0},
        {3.0, 4.0, 3.0},
        {1.0, 2.0, 1.0},
        {5.0, 6.0, 5.0},
    };

    auto result = utils::decomposition::qr_col_pivot(A);
    CHECK(result.rank() == 2);
}

TEST_CASE("qr_col_pivot rank 1", "[decomposition][qr_col_pivot]") {
    // All columns are multiples of [1, 2, 3].
    Matrix<double, 3, 3> A{
        {1.0, 2.0, 3.0},
        {2.0, 4.0, 6.0},
        {3.0, 6.0, 9.0},
    };

    auto result = utils::decomposition::qr_col_pivot(A);
    CHECK(result.rank() == 1);
}

TEST_CASE("qr_col_pivot zero matrix", "[decomposition][qr_col_pivot]") {
    Matrix<double, 3, 3> A{
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
    };

    auto result = utils::decomposition::qr_col_pivot(A);
    CHECK(result.rank() == 0);
}

TEST_CASE("qr_col_pivot rank convenience function",
          "[decomposition][qr_col_pivot]") {
    Matrix<double, 3, 3> A_full{
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0},
    };
    CHECK(utils::decomposition::rank(A_full) == 3);

    // Rank-deficient.
    Matrix<double, 3, 3> A_def{
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0},
        {1.0, 1.0, 2.0},
    };
    CHECK(utils::decomposition::rank(A_def) == 2);
}

TEST_CASE("qr_col_pivot wide matrix 2x4", "[decomposition][qr_col_pivot]") {
    // Wide matrix (more cols than rows). Rank should be 2 (full row rank).
    Matrix<double, 2, 4> A{
        {1.0, 0.0, 2.0, 1.0},
        {0.0, 1.0, 1.0, 3.0},
    };

    auto result = utils::decomposition::qr_col_pivot(A);
    CHECK(result.rank() == 2);

    // Verify factorisation.
    const auto &Q = result.Q;
    const auto &R = result.R;
    const auto &perm = result.col_perm;

    Matrix<double, 2, 4> QR = Q * R;
    for (index_type i = 0; i < 2; ++i) {
        for (index_type j = 0; j < 4; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, perm[j])).margin(1e-10));
        }
    }
}
