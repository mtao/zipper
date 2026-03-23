#if !defined(ZIPPER_STORAGE_DETAIL_SPARSECOMPRESSEDDATA_HPP)
#define ZIPPER_STORAGE_DETAIL_SPARSECOMPRESSEDDATA_HPP

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <zipper/concepts/Index.hpp>
#include <zipper/types.hpp>
namespace zipper::storage::detail {

// ═══════════════════════════════════════════════════════════════════════
// Base case: rank-1 (vectors) or the leaf level of higher-rank tensors.
//
// Stores nonzero entries as two parallel arrays:
//   m_indices[k] — the sorted index of the k-th nonzero
//   m_values[k]  — the corresponding value
//
// This is SOA (Structure-of-Arrays) layout, enabling zero-copy interop
// with external sparse libraries (SuiteSparse, Eigen, etc.).
// ═══════════════════════════════════════════════════════════════════════

template <typename T, rank_type N>
  requires(N >= 0)
struct SparseCompressedData;

template <typename T> struct SparseCompressedData<T, 0> {
  using value_type = T;

  std::vector<index_type> m_indices = {};
  std::vector<T> m_values = {};

  SparseCompressedData() = default;
  SparseCompressedData(SparseCompressedData &&) = default;
  auto operator=(SparseCompressedData &&) -> SparseCompressedData & = default;
  SparseCompressedData(const SparseCompressedData &) = default;
  auto operator=(const SparseCompressedData &)
      -> SparseCompressedData & = default;
  ~SparseCompressedData();

  // ── Size / capacity ─────────────────────────────────────────────────
  [[nodiscard]] auto size() const -> size_t { return m_indices.size(); }
  [[nodiscard]] auto nnz() const -> size_t { return m_indices.size(); }

  auto indices_data() -> index_type * { return m_indices.data(); }
  auto indices_data() const -> const index_type * { return m_indices.data(); }
  auto values_data() -> T * { return m_values.data(); }
  auto values_data() const -> const T * { return m_values.data(); }

  void reserve(size_t n) {
    m_indices.reserve(n);
    m_values.reserve(n);
  }

  // ── Element access ──────────────────────────────────────────────────
  auto __attribute__((pure)) coeff_(size_t start, size_t size,
                                    index_type index) const -> value_type {
    auto begin = m_indices.begin() + start;
    auto end = begin + size;
    auto it = std::lower_bound(begin, end, index);
    if (it != end && *it == index) {
      auto pos = static_cast<size_t>(it - m_indices.begin());
      return m_values[pos];
    }
    return value_type(0);
  }

  auto __attribute__((pure)) coeff(index_type index) const -> value_type {
    return coeff_(0, m_indices.size(), index);
  }

  auto coeff_ref_(size_t start, size_t size, index_type index)
      -> value_type & {
    auto begin = m_indices.begin() + start;
    auto end = begin + size;
    auto it = std::lower_bound(begin, end, index);
    if (it != end && *it == index) {
      auto pos = static_cast<size_t>(it - m_indices.begin());
      return m_values[pos];
    }
    throw std::invalid_argument("Index not found in SparseCompressedData");
  }

  auto coeff_ref(index_type index) -> value_type & {
    return coeff_ref_(0, m_indices.size(), index);
  }

  auto const_coeff_ref_(size_t start, size_t size, index_type index) const
      -> const value_type & {
    auto begin = m_indices.begin() + start;
    auto end = begin + size;
    auto it = std::lower_bound(begin, end, index);
    if (it != end && *it == index) {
      auto pos = static_cast<size_t>(it - m_indices.begin());
      return m_values[pos];
    }
    throw std::invalid_argument("Index not found in SparseCompressedData");
  }

  auto const_coeff_ref(index_type index) const -> const value_type & {
    return const_coeff_ref_(0, m_indices.size(), index);
  }

  // ── Insertion (append-only, sorted order) ───────────────────────────
  auto insert_back_(size_t /*start*/, size_t /*size*/, index_type index)
      -> value_type & {
    m_indices.push_back(index);
    m_values.push_back(value_type(0));
    return m_values.back();
  }

  auto insert_back(index_type index) -> value_type & {
    return insert_back_(0, m_indices.size(), index);
  }
};

// ═══════════════════════════════════════════════════════════════════════
// Recursive case: rank-2+ (matrices, 3-tensors, etc.).
//
// Adds a dense outer-pointer array `m_indptr` of size (n_outer + 1).
// m_indptr[j] is the offset into the parent level's arrays where
// outer index j begins. m_indptr[n_outer] == parent.nnz().
//
// For a rank-2 matrix (N=1):
//   m_indptr   — column pointers (CSC) or row pointers (CSR)
//   inherited m_indices — row indices (CSC) or column indices (CSR)
//   inherited m_values  — nonzero values
//
// This matches the standard CSR/CSC format used by SuiteSparse, Eigen,
// scipy, and every other sparse library.
// ═══════════════════════════════════════════════════════════════════════

template <typename T, rank_type N>
  requires(N >= 0)
struct SparseCompressedData : public SparseCompressedData<T, N - 1> {
  using Parent = SparseCompressedData<T, N - 1>;
  using value_type = T;

  // Dense outer pointer array. Size = n_outer + 1, or empty if not yet
  // initialized (e.g., when constructed via insert_back without
  // resize_outer).
  std::vector<index_type> m_indptr = {};

  SparseCompressedData() = default;
  SparseCompressedData(SparseCompressedData &&) = default;
  auto operator=(SparseCompressedData &&) -> SparseCompressedData & = default;
  SparseCompressedData(const SparseCompressedData &) = default;
  auto operator=(const SparseCompressedData &)
      -> SparseCompressedData & = default;
  ~SparseCompressedData();

  /// Initialize the outer pointer array to represent `n_outer` outer
  /// indices, all initially empty (indptr = all zeros).
  void resize_outer(index_type n_outer) {
    m_indptr.assign(n_outer + 1, index_type(0));
  }

  /// Number of outer indices (0 if not initialized).
  [[nodiscard]] auto outer_size() const -> index_type {
    return m_indptr.empty() ? 0 : m_indptr.size() - 1;
  }

  auto indptr_data() -> index_type * { return m_indptr.data(); }
  auto indptr_data() const -> const index_type * { return m_indptr.data(); }

  [[nodiscard]] auto size() const -> size_t {
    return m_indptr.empty() ? 0 : m_indptr.size() - 1;
  }

  // ── Element access ──────────────────────────────────────────────────
  template <zipper::concepts::IndexPack... Leftover>
    requires(sizeof...(Leftover) == N)
  auto __attribute__((pure)) coeff_(size_t /*start*/, size_t /*size*/, index_type index,
                                    Leftover... leftover) const -> value_type {
    if (m_indptr.empty()) {
      return value_type(0);
    }
    if (index >= m_indptr.size() - 1) {
      return value_type(0);
    }
    auto row_start = m_indptr[index];
    auto row_end = m_indptr[index + 1];
    if (row_start == row_end) {
      return value_type(0);
    }
    return Parent::coeff_(row_start, row_end - row_start, leftover...);
  }

  template <typename... Leftover>
    requires((sizeof...(Leftover) == N) &&
             zipper::concepts::IndexPack<Leftover...>)
  auto __attribute__((pure)) coeff(index_type index, Leftover... leftover) const
      -> value_type {
    return coeff_(0, 0, index, leftover...);
  }

  template <zipper::concepts::IndexPack... Leftover>
    requires(sizeof...(Leftover) == N)
  auto coeff_ref_(size_t /*start*/, size_t /*size*/, index_type index,
                  Leftover... leftover) -> value_type & {
    if (m_indptr.empty() || index >= m_indptr.size() - 1) {
      throw std::invalid_argument("Index not found in SparseCompressedData");
    }
    auto row_start = m_indptr[index];
    auto row_end = m_indptr[index + 1];
    if (row_start == row_end) {
      throw std::invalid_argument("Index not found in SparseCompressedData");
    }
    return Parent::coeff_ref_(row_start, row_end - row_start, leftover...);
  }

  template <typename... Leftover>
    requires((sizeof...(Leftover) == N) &&
             zipper::concepts::IndexPack<Leftover...>)
  auto coeff_ref(index_type index, Leftover... leftover) -> value_type & {
    return coeff_ref_(0, 0, index, leftover...);
  }

  template <zipper::concepts::IndexPack... Leftover>
    requires(sizeof...(Leftover) == N)
  auto const_coeff_ref_(size_t /*start*/, size_t /*size*/, index_type index,
                        Leftover... leftover) const -> const value_type & {
    if (m_indptr.empty() || index >= m_indptr.size() - 1) {
      throw std::invalid_argument("Index not found in SparseCompressedData");
    }
    auto row_start = m_indptr[index];
    auto row_end = m_indptr[index + 1];
    if (row_start == row_end) {
      throw std::invalid_argument("Index not found in SparseCompressedData");
    }
    return Parent::const_coeff_ref_(row_start, row_end - row_start,
                                    leftover...);
  }

  template <typename... Leftover>
    requires((sizeof...(Leftover) == N) &&
             zipper::concepts::IndexPack<Leftover...>)
  auto const_coeff_ref(index_type index, Leftover... leftover) const
      -> const value_type & {
    return const_coeff_ref_(0, 0, index, leftover...);
  }

  // ── Insertion (append-only, sorted outer index order) ───────────────
  //
  // insert_back maintains the m_indptr invariant: when appending an entry
  // for outer index `index`, all indptr[index+1 .. end] are incremented.
  //
  // This is O(n_outer) per insertion in the worst case, but the typical
  // use pattern is bulk construction via to_sparse_compressed_data which
  // builds indptr directly (O(nnz) total). insert_back is provided for
  // convenience in tests and ad-hoc construction.
  template <zipper::concepts::IndexPack... Leftover>
    requires(sizeof...(Leftover) == N)
  auto insert_back(index_type index, Leftover... leftover) -> value_type & {
    // Enforce sorted insertion order: outer index must be non-decreasing.
    if (!m_indptr.empty() && m_indptr.size() >= 2) {
      // Find the last outer index that had entries
      // (i.e., the largest j such that indptr[j+1] > indptr[j])
      index_type last_outer = m_indptr.size() - 2;
      // Walk back to find the actual last used outer
      while (last_outer > 0 && m_indptr[last_outer + 1] == m_indptr[last_outer]) {
        last_outer--;
      }
      // If there are any entries, the last used outer is where the data ends
      if (m_indptr[last_outer + 1] > m_indptr[last_outer] && index < last_outer) {
        throw std::invalid_argument("Inserted index out of order");
      }
    }

    // Ensure m_indptr is large enough for this outer index
    if (m_indptr.empty()) {
      // Not yet sized — grow to accommodate
      m_indptr.resize(index + 2, index_type(0));
    } else if (index >= m_indptr.size() - 1) {
      // Grow m_indptr
      m_indptr.resize(index + 2, m_indptr.back());
    }

    // All indptr entries after `index` must be incremented
    for (size_t k = index + 1; k < m_indptr.size(); ++k) {
      m_indptr[k]++;
    }

    // Delegate to parent. For sorted-order insertion (which is the contract
    // of insert_back), appending to the parent's arrays is correct because
    // entries are naturally in order.
    return Parent::insert_back(leftover...);
  }
};

template <typename T, rank_type N>
  requires(N >= 0)
SparseCompressedData<T, N>::~SparseCompressedData() = default;
template <typename T>
SparseCompressedData<T, 0>::~SparseCompressedData() = default;
} // namespace zipper::storage::detail

#endif
