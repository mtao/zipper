#include <concepts>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/decomposition/ldlt.hpp>
#include <zipper/utils/decomposition/llt.hpp>
#include <zipper/utils/decomposition/lu.hpp>
#include <zipper/utils/decomposition/polar.hpp>
#include <zipper/utils/decomposition/qr.hpp>
#include <zipper/utils/decomposition/svd.hpp>

#include "catch_include.hpp"

using namespace zipper;
using namespace zipper::utils::decomposition;
namespace solver = zipper::utils::solver;

namespace test {

struct FieldScalar {
    long double value = 0;

    constexpr FieldScalar() = default;
    constexpr FieldScalar(int input) : value(input) {}
    constexpr explicit FieldScalar(long double input) : value(input) {}

    constexpr auto operator+=(FieldScalar rhs) -> FieldScalar & {
        value += rhs.value;
        return *this;
    }
    constexpr auto operator-=(FieldScalar rhs) -> FieldScalar & {
        value -= rhs.value;
        return *this;
    }
    constexpr auto operator*=(FieldScalar rhs) -> FieldScalar & {
        value *= rhs.value;
        return *this;
    }
    constexpr auto operator/=(FieldScalar rhs) -> FieldScalar & {
        value /= rhs.value;
        return *this;
    }

    friend constexpr auto operator+(FieldScalar lhs, FieldScalar rhs)
        -> FieldScalar {
        return lhs += rhs;
    }
    friend constexpr auto operator-(FieldScalar lhs, FieldScalar rhs)
        -> FieldScalar {
        return lhs -= rhs;
    }
    friend constexpr auto operator*(FieldScalar lhs, FieldScalar rhs)
        -> FieldScalar {
        return lhs *= rhs;
    }
    friend constexpr auto operator/(FieldScalar lhs, FieldScalar rhs)
        -> FieldScalar {
        return lhs /= rhs;
    }
    friend constexpr auto operator-(FieldScalar value) -> FieldScalar {
        return FieldScalar{-value.value};
    }
    friend constexpr auto operator<=>(FieldScalar, FieldScalar) = default;
};

inline auto abs(FieldScalar value) -> FieldScalar {
    return FieldScalar{std::abs(value.value)};
}

inline auto sqrt(FieldScalar value) -> FieldScalar {
    return FieldScalar{std::sqrt(value.value)};
}

} // namespace test

template <>
struct zipper::concepts::detail::HasDivision<test::FieldScalar>
    : std::true_type {};

template <>
struct zipper::utils::scalar_math::traits<test::FieldScalar> {
    static constexpr auto epsilon() noexcept -> test::FieldScalar {
        return test::FieldScalar{std::numeric_limits<long double>::epsilon()};
    }
};

template <typename MatrixType>
concept HasLlt = requires(const MatrixType &matrix) { llt(matrix); };

template <typename MatrixType>
concept HasLdlt = requires(const MatrixType &matrix) { ldlt(matrix); };

template <typename MatrixType>
concept HasPlu = requires(const MatrixType &matrix) { plu(matrix); };

template <typename MatrixType>
concept HasPolar = requires(const MatrixType &matrix) { polar(matrix); };

template <typename MatrixType>
concept HasProperPolar = requires(const MatrixType &matrix) {
    proper_polar(matrix);
};

template <typename MatrixType>
concept HasQr = requires(const MatrixType &matrix) {
    qr(matrix);
    qr_full(matrix);
    qr_gram_schmidt(matrix);
    qr_col_pivot(matrix);
};

template <typename MatrixType>
concept HasSvd = requires(const MatrixType &matrix) { svd(matrix); };

template <typename MatrixType, typename VectorType>
concept HasQrSolve = requires(const MatrixType &matrix, const VectorType &rhs) {
    qr_solve(matrix, rhs);
    qr_solve_full(matrix, rhs);
};

template <typename Result>
concept HasRvalueL = requires(Result result) { std::move(result).L(); };

template <typename Result>
concept HasRvalueU = requires(Result result) { std::move(result).U(); };

static_assert(!HasLlt<Matrix<double, 2, 3>>);
static_assert(!HasLdlt<Matrix<double, 2, 3>>);
static_assert(!HasPlu<Matrix<double, 2, 3>>);
static_assert(!HasPolar<Matrix<double, 2, 3>>);
static_assert(!HasProperPolar<Matrix<double, 2, 3>>);
static_assert(!HasSvd<Matrix<int, 2, 2>>);
static_assert(!HasPolar<Matrix<int, 2, 2>>);
static_assert(!HasProperPolar<Matrix<int, 2, 2>>);
static_assert(HasSvd<Matrix<float, 2, 2>>);
static_assert(HasPolar<Matrix<float, 2, 2>>);
static_assert(HasProperPolar<Matrix<float, 2, 2>>);
static_assert(HasQr<Matrix<test::FieldScalar, 2, 2>>);
static_assert(HasPolar<Matrix<test::FieldScalar, 2, 2>>);
static_assert(HasProperPolar<Matrix<test::FieldScalar, 2, 2>>);
static_assert(!HasQr<Matrix<int, 2, 2>>);
static_assert(std::same_as<decltype(svd(std::declval<Matrix<float, 2, 2>>())),
                           SVDResult<float, 2, 2>>);
static_assert(!HasQrSolve<Matrix<double, 2, 3>, Vector<double, 2>>);
static_assert(!HasQrSolve<Matrix<double, 3, 2>, Vector<double, 2>>);
static_assert(!HasRvalueL<PLUResult<double, 2>>);
static_assert(!HasRvalueU<PLUResult<double, 2>>);

TEST_CASE("orthogonal_decompositions_support_custom_field_scalars",
          "[decomposition][contracts][scalar]") {
    using T = test::FieldScalar;
    Vector<T, 2> vector{T{3}, T{4}};
    CHECK(vector.norm().value == Catch::Approx(5.0L));

    Matrix<T, 2, 2> matrix{{T{3}, T{1}}, {T{0}, T{2}}};
    const auto qr_result = qr(matrix);
    const Matrix<T, 2, 2> qr_reconstructed(qr_result.Q * qr_result.R);
    const auto polar_result = polar(matrix);
    const Matrix<T, 2, 2> polar_reconstructed(
        polar_result.R * polar_result.S);

    for (index_type i = 0; i < 2; ++i) {
        for (index_type j = 0; j < 2; ++j) {
            CHECK(qr_reconstructed(i, j).value ==
                  Catch::Approx(matrix(i, j).value).margin(1e-15L));
            CHECK(polar_reconstructed(i, j).value ==
                  Catch::Approx(matrix(i, j).value).margin(1e-15L));
        }
    }
}
static_assert(HasLlt<Matrix<double, 2, dynamic_extent>>);
static_assert(HasLlt<Matrix<double, dynamic_extent, 2>>);
static_assert(HasQrSolve<Matrix<double, 3, dynamic_extent>, Vector<double, 3>>);

TEST_CASE("square_decompositions_reject_dynamic_nonsquare_matrices",
          "[decomposition][contracts]") {
    MatrixXX<double> matrix(2, 3);

    const auto llt_result = llt(matrix);
    const auto ldlt_result = ldlt(matrix);
    const auto plu_result = plu(matrix);

    REQUIRE_FALSE(llt_result);
    REQUIRE_FALSE(ldlt_result);
    REQUIRE_FALSE(plu_result);
    CHECK(llt_result.error().kind == solver::SolverError::Kind::invalid_input);
    CHECK(ldlt_result.error().kind == solver::SolverError::Kind::invalid_input);
    CHECK(plu_result.error().kind == solver::SolverError::Kind::invalid_input);
}

TEST_CASE("square_decompositions_reject_partially_dynamic_nonsquare_matrices",
          "[decomposition][contracts]") {
    Matrix<double, 2, dynamic_extent> static_rows(3);
    Matrix<double, dynamic_extent, 2> static_cols(3);

    CHECK(llt(static_rows).error().kind ==
          solver::SolverError::Kind::invalid_input);
    CHECK(ldlt(static_rows).error().kind ==
          solver::SolverError::Kind::invalid_input);
    CHECK(plu(static_rows).error().kind ==
          solver::SolverError::Kind::invalid_input);
    CHECK(llt(static_cols).error().kind ==
          solver::SolverError::Kind::invalid_input);
}

TEST_CASE("decomposition_free_solves_reject_dynamic_rhs_mismatches",
          "[decomposition][contracts]") {
    MatrixXX<double> matrix{{4.0, 1.0}, {1.0, 3.0}};
    VectorX<double> rhs{1.0, 2.0, 3.0};

    const auto llt_result = llt_solve(matrix, rhs);
    const auto ldlt_result = ldlt_solve(matrix, rhs);
    const auto plu_result = plu_solve(matrix, rhs);

    REQUIRE_FALSE(llt_result);
    REQUIRE_FALSE(ldlt_result);
    REQUIRE_FALSE(plu_result);
    CHECK(llt_result.error().kind == solver::SolverError::Kind::invalid_input);
    CHECK(ldlt_result.error().kind == solver::SolverError::Kind::invalid_input);
    CHECK(plu_result.error().kind == solver::SolverError::Kind::invalid_input);
}

TEST_CASE("stored_decomposition_solves_reject_dynamic_rhs_mismatches",
          "[decomposition][contracts]") {
    MatrixXX<double> matrix{{4.0, 1.0}, {1.0, 3.0}};
    VectorX<double> rhs{1.0, 2.0, 3.0};
    auto llt_result = llt(matrix);
    auto ldlt_result = ldlt(matrix);
    auto plu_result = plu(matrix);
    REQUIRE(llt_result);
    REQUIRE(ldlt_result);
    REQUIRE(plu_result);

    const auto llt_solve_result = llt_result->solve(rhs);
    const auto ldlt_solve_result = ldlt_result->solve(rhs);
    const auto plu_solve_result = plu_result->solve(rhs);
    REQUIRE_FALSE(llt_solve_result);
    REQUIRE_FALSE(ldlt_solve_result);
    REQUIRE_FALSE(plu_solve_result);
    CHECK(llt_solve_result.error().kind ==
          solver::SolverError::Kind::invalid_input);
    CHECK(ldlt_solve_result.error().kind ==
          solver::SolverError::Kind::invalid_input);
    CHECK(plu_solve_result.error().kind ==
          solver::SolverError::Kind::invalid_input);
}

TEST_CASE("qr_factors_wide_matrices_but_rejects_solving_them",
          "[decomposition][qr][contracts]") {
    MatrixXX<double> matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 7.0}};
    VectorX<double> rhs{1.0, 2.0};

    const auto reduced = qr(matrix);
    const auto full = qr_full(matrix);
    CHECK(reduced.Q.extent(0) == 2);
    CHECK(reduced.R.extent(1) == 3);
    CHECK(full.Q.extent(0) == 2);
    CHECK(full.R.extent(1) == 3);

    const auto reduced_result = reduced.solve(rhs);
    const auto full_result = full.solve(rhs);
    const auto free_result = qr_solve(matrix, rhs);
    const auto free_full_result = qr_solve_full(matrix, rhs);
    REQUIRE_FALSE(reduced_result);
    REQUIRE_FALSE(full_result);
    REQUIRE_FALSE(free_result);
    REQUIRE_FALSE(free_full_result);
    CHECK(reduced_result.error().kind ==
          solver::SolverError::Kind::invalid_input);
    CHECK(full_result.error().kind == solver::SolverError::Kind::invalid_input);
    CHECK(free_result.error().kind == solver::SolverError::Kind::invalid_input);
    CHECK(free_full_result.error().kind ==
          solver::SolverError::Kind::invalid_input);
}

TEST_CASE("qr_solves_reject_dynamic_rhs_mismatches",
          "[decomposition][qr][contracts]") {
    MatrixXX<double> matrix{{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    VectorX<double> rhs{1.0, 2.0};

    const auto reduced_result = qr(matrix).solve(rhs);
    const auto full_result = qr_full(matrix).solve(rhs);
    const auto free_result = qr_solve(matrix, rhs);
    const auto free_full_result = qr_solve_full(matrix, rhs);
    REQUIRE_FALSE(reduced_result);
    REQUIRE_FALSE(full_result);
    REQUIRE_FALSE(free_result);
    REQUIRE_FALSE(free_full_result);
    CHECK(reduced_result.error().kind ==
          solver::SolverError::Kind::invalid_input);
    CHECK(full_result.error().kind == solver::SolverError::Kind::invalid_input);
    CHECK(free_result.error().kind == solver::SolverError::Kind::invalid_input);
    CHECK(free_full_result.error().kind ==
          solver::SolverError::Kind::invalid_input);
}

TEST_CASE("reduced_qr_solve_rejects_malformed_dynamic_factors",
          "[decomposition][qr][contracts]") {
    QRReducedResult<double, dynamic_extent, dynamic_extent> factors{
        .Q = MatrixXX<double>(3, 2),
        .R = MatrixXX<double>(2, 1)};
    VectorX<double> rhs{1.0, 2.0, 3.0};

    const auto result = factors.solve(rhs);

    REQUIRE_FALSE(result);
    CHECK(result.error().kind == solver::SolverError::Kind::invalid_input);
}

TEST_CASE("polar_decompositions_reject_a_dynamic_nonsquare_matrix",
           "[decomposition][polar][contracts]") {
    MatrixXX<double> matrix(2, 3);
    CHECK_THROWS_AS(polar(matrix), std::invalid_argument);
    CHECK_THROWS_AS(proper_polar(matrix), std::invalid_argument);
}
