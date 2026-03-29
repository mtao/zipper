#if !defined(ZIPPER_SPARSEENTRY_HPP)
#define ZIPPER_SPARSEENTRY_HPP

/// @file SparseEntry.hpp
/// @brief Sparse entry (triplet) type for constructing sparse matrices/vectors.
/// @ingroup user_types
///
/// `SparseEntry<T, D>` represents a single non-zero entry in a sparse
/// tensor of rank `D`.  It stores `D` indices and one scalar value.
///
/// @code
///   // Matrix entries (rank-2)
///   SparseEntry<double, 2> e{.indices = {0, 1}, .value = 3.14};
///
///   // Vector entries (rank-1)
///   SparseEntry<double, 1> e{.indices = {5}, .value = 2.0};
///
///   // Construct from a range
///   std::vector<SparseEntry<double, 2>> entries = {
///       {{0, 0}, 1.0}, {{1, 1}, 2.0}, {{2, 2}, 3.0}};
///   COOMatrix<double, 3, 3> A(entries);
/// @endcode

#include <array>
#include "zipper/types.hpp"

namespace zipper {

/// A single non-zero entry in a sparse tensor of rank `D`.
template <typename T, rank_type D>
struct SparseEntry {
    std::array<index_type, D> indices;
    T value;
};

} // namespace zipper

#endif
