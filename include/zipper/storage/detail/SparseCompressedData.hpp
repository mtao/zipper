#if !defined(ZIPPER_STORAGE_DETAIL_SPARSECOMPRESSEDDATA_HPP)
#define ZIPPER_STORAGE_DETAIL_SPARSECOMPRESSEDDATA_HPP

#include <algorithm>
#include <cassert>
#include <span>
#include <stdexcept>
#include <vector>
#include <zipper/concepts/Index.hpp>
#include <zipper/types.hpp>
namespace zipper::storage::detail {

// ═══════════════════════════════════════════════════════════════════════
// Storage policy tags
//
// OwnedStorage — data is stored in std::vector (owning, resizable).
// SpanStorage  — data is stored in std::span (non-owning, fixed).
//
// SparseCompressedData is parameterized on a StoragePolicy tag which
// selects the container type for indices, values, and indptr arrays.
// The default is OwnedStorage, preserving backward compatibility.
// ═══════════════════════════════════════════════════════════════════════

struct OwnedStorage {};
struct SpanStorage {};

/// True when S is a non-owning span-based storage policy.
template <typename S>
constexpr bool is_span_storage_v = std::is_same_v<S, SpanStorage>;

/// Select the container type for a given storage policy and element type.
///   OwnedStorage → std::vector<T>
///   SpanStorage  → std::span<T>   (T may be const-qualified)
template <typename StoragePolicy, typename T> struct StorageContainer;

template <typename T> struct StorageContainer<OwnedStorage, T> {
  using type = std::vector<std::remove_const_t<T>>;
};
template <typename T> struct StorageContainer<SpanStorage, T> {
  using type = std::span<T>;
};

template <typename StoragePolicy, typename T>
using storage_container_t = typename StorageContainer<StoragePolicy, T>::type;

// ═══════════════════════════════════════════════════════════════════════
// Base case: rank-1 (vectors) or the leaf level of higher-rank tensors.
//
// Stores nonzero entries as two parallel arrays:
//   m_indices[k] — the sorted index of the k-th nonzero
//   m_values[k]  — the corresponding value
//
// This is SOA (Structure-of-Arrays) layout, enabling zero-copy interop
// with external sparse libraries (SuiteSparse, Eigen, etc.).
//
// When StoragePolicy = SpanStorage, the containers are non-owning spans
// and mutation methods (insert_back, reserve) are constrained away.
// ═══════════════════════════════════════════════════════════════════════

template <typename T, rank_type N, typename StoragePolicy = OwnedStorage>
  requires(N >= 0)
struct SparseCompressedData;

template <typename T, typename StoragePolicy>
struct SparseCompressedData<T, 0, StoragePolicy> {
  using value_type = T;
  using storage_policy = StoragePolicy;
  static constexpr bool is_span = is_span_storage_v<StoragePolicy>;

  // For span storage, indices are always const (structural immutability).
  // Values may be mutable or const depending on T's const qualification.
  using index_container_type =
      storage_container_t<StoragePolicy, const index_type>;
  using value_container_type = storage_container_t<StoragePolicy, T>;

  index_container_type m_indices = {};
  value_container_type m_values = {};

  SparseCompressedData() = default;
  SparseCompressedData(SparseCompressedData &&) = default;
  auto operator=(SparseCompressedData &&) -> SparseCompressedData & = default;
  SparseCompressedData(const SparseCompressedData &) = default;
  auto operator=(const SparseCompressedData &)
      -> SparseCompressedData & = default;
  ~SparseCompressedData();

  /// Construct from spans (span storage only).
  SparseCompressedData(std::span<const index_type> indices,
                       std::span<T> values)
    requires(is_span)
      : m_indices(indices), m_values(values) {}

  // ── Size / capacity ─────────────────────────────────────────────────
  [[nodiscard]] auto size() const -> size_t { return m_indices.size(); }
  [[nodiscard]] auto nnz() const -> size_t { return m_indices.size(); }

  auto indices_data() const -> const index_type * { return m_indices.data(); }
  auto values_data() const -> const T * { return m_values.data(); }

  auto indices_data() -> const index_type *
    requires(is_span)
  {
    return m_indices.data();
  }
  auto indices_data() -> index_type *
    requires(!is_span)
  {
    return m_indices.data();
  }
  auto values_data() -> T * { return m_values.data(); }

  void reserve(size_t n)
    requires(!is_span)
  {
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
      -> value_type &
    requires(!is_span)
  {
    m_indices.push_back(index);
    m_values.push_back(value_type(0));
    return m_values.back();
  }

  auto insert_back(index_type index) -> value_type &
    requires(!is_span)
  {
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
//
// When StoragePolicy = SpanStorage, mutation methods (insert_back,
// resize_outer) are constrained away.
// ═══════════════════════════════════════════════════════════════════════

template <typename T, rank_type N, typename StoragePolicy>
  requires(N >= 0)
struct SparseCompressedData
    : public SparseCompressedData<T, N - 1, StoragePolicy> {
  using Parent = SparseCompressedData<T, N - 1, StoragePolicy>;
  using value_type = T;
  using storage_policy = StoragePolicy;
  static constexpr bool is_span = is_span_storage_v<StoragePolicy>;

  // indptr is always structurally immutable for span storage.
  using indptr_container_type =
      storage_container_t<StoragePolicy, const index_type>;

  // Dense outer pointer array. Size = n_outer + 1, or empty if not yet
  // initialized (e.g., when constructed via insert_back without
  // resize_outer).
  indptr_container_type m_indptr = {};

  SparseCompressedData() = default;
  SparseCompressedData(SparseCompressedData &&) = default;
  auto operator=(SparseCompressedData &&) -> SparseCompressedData & = default;
  SparseCompressedData(const SparseCompressedData &) = default;
  auto operator=(const SparseCompressedData &)
      -> SparseCompressedData & = default;
  ~SparseCompressedData();

  /// Construct from spans (span storage, rank-2 only via N==1).
  SparseCompressedData(std::span<const index_type> indptr,
                       std::span<const index_type> indices,
                       std::span<T> values)
    requires(is_span && N == 1)
      : Parent(indices, values), m_indptr(indptr) {}

  /// Initialize the outer pointer array to represent `n_outer` outer
  /// indices, all initially empty (indptr = all zeros).
  void resize_outer(index_type n_outer)
    requires(!is_span)
  {
    m_indptr.assign(n_outer + 1, index_type(0));
  }

  /// Number of outer indices (0 if not initialized).
  [[nodiscard]] auto outer_size() const -> index_type {
    return m_indptr.empty() ? 0 : m_indptr.size() - 1;
  }

  auto indptr_data() const -> const index_type * { return m_indptr.data(); }
  auto indptr_data() -> const index_type *
    requires(is_span)
  {
    return m_indptr.data();
  }
  auto indptr_data() -> index_type *
    requires(!is_span)
  {
    return m_indptr.data();
  }

  [[nodiscard]] auto size() const -> size_t {
    return m_indptr.empty() ? 0 : m_indptr.size() - 1;
  }

  // ── Element access ──────────────────────────────────────────────────
  template <zipper::concepts::IndexPack... Leftover>
    requires(sizeof...(Leftover) == N)
  auto __attribute__((pure)) coeff_(size_t /*start*/, size_t /*size*/,
                                    index_type index,
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
  auto __attribute__((pure)) coeff(index_type index,
                                   Leftover... leftover) const -> value_type {
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
  //
  // Only available for owned storage.
  template <zipper::concepts::IndexPack... Leftover>
    requires(sizeof...(Leftover) == N && !is_span)
  auto insert_back(index_type index, Leftover... leftover) -> value_type & {
    // Enforce sorted insertion order: outer index must be non-decreasing.
    if (!m_indptr.empty() && m_indptr.size() >= 2) {
      // Find the last outer index that had entries
      // (i.e., the largest j such that indptr[j+1] > indptr[j])
      index_type last_outer = m_indptr.size() - 2;
      // Walk back to find the actual last used outer
      while (last_outer > 0 &&
             m_indptr[last_outer + 1] == m_indptr[last_outer]) {
        last_outer--;
      }
      // If there are any entries, the last used outer is where the data ends
      if (m_indptr[last_outer + 1] > m_indptr[last_outer] &&
          index < last_outer) {
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

template <typename T, rank_type N, typename StoragePolicy>
  requires(N >= 0)
SparseCompressedData<T, N, StoragePolicy>::~SparseCompressedData() = default;
template <typename T, typename StoragePolicy>
SparseCompressedData<T, 0, StoragePolicy>::~SparseCompressedData() = default;

// ═══════════════════════════════════════════════════════════════════════
// Convenience alias: the owned (default) variant.
// ═══════════════════════════════════════════════════════════════════════

/// Non-owning span-based view of compressed sparse data (CSR/CSC).
///
/// Constructed from raw pointers or spans into externally owned memory.
/// Mutation of values is possible when T is non-const; structural
/// mutation (insert_back, resize_outer) is not available.
template <typename T, rank_type N>
using SparseCompressedSpanData = SparseCompressedData<T, N, SpanStorage>;

} // namespace zipper::storage::detail

#endif
