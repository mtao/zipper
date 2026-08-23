#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <utility>

#include <zipper/types.hpp>
#include <zipper/utils/decomposition/detail/shape_concepts.hpp>
#include <zipper/utils/solver/error.hpp>

namespace zipper::utils::decomposition::detail {

inline auto invalid_shape(std::string message) -> solver::SolverError {
    return {.kind = solver::SolverError::Kind::invalid_input,
            .message = std::move(message)};
}

template <typename MatrixType>
auto validate_square(const MatrixType &matrix, std::string_view operation)
    -> std::expected<void, solver::SolverError> {
    if (matrix.extent(0) != matrix.extent(1)) {
        return std::unexpected(invalid_shape(
            std::string(operation) + ": matrix must be square"));
    }
    return {};
}

template <typename VectorType>
auto validate_rhs(index_type rows, const VectorType &rhs,
                  std::string_view operation)
    -> std::expected<void, solver::SolverError> {
    if (rhs.extent(0) != rows) {
        return std::unexpected(invalid_shape(
            std::string(operation) + ": right-hand side size must match rows"));
    }
    return {};
}

template <typename MatrixType>
auto validate_not_wide(const MatrixType &matrix, std::string_view operation)
    -> std::expected<void, solver::SolverError> {
    if (matrix.extent(0) < matrix.extent(1)) {
        return std::unexpected(invalid_shape(
            std::string(operation) + ": solving wide systems is unsupported"));
    }
    return {};
}

template <typename Left, typename Right>
auto validate_equal_extent(const Left &left, index_type left_axis,
                           const Right &right, index_type right_axis,
                           std::string_view operation)
    -> std::expected<void, solver::SolverError> {
    if (left.extent(left_axis) != right.extent(right_axis)) {
        return std::unexpected(invalid_shape(
            std::string(operation) + ": factor dimensions are inconsistent"));
    }
    return {};
}

} // namespace zipper::utils::decomposition::detail
