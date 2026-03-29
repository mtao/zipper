/// @file error.hpp
/// @brief Error type for sparse matrix operations.
/// @ingroup sparse_ops

#if !defined(ZIPPER_UTILS_SPARSE_ERROR_HPP)
#define ZIPPER_UTILS_SPARSE_ERROR_HPP

#include <string>

namespace zipper::utils::sparse {

/// Describes what went wrong in a sparse matrix operation.
///
/// Failure modes:
///   - `dimension_mismatch`: operand dimensions are incompatible.
struct SparseError {
    enum class Kind {
        dimension_mismatch,
    };

    Kind kind;
    std::string message;
};

} // namespace zipper::utils::sparse

#endif
