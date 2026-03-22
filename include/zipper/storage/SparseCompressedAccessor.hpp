#if !defined(ZIPPER_STORAGE_SPARSECOMPRESSEDACCESSOR_HPP)
#define ZIPPER_STORAGE_SPARSECOMPRESSEDACCESSOR_HPP

#include <stdexcept>
#include <vector>

#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/expression/ExpressionBase.hpp"
#include "zipper/expression/detail/IndexSet.hpp"
#include "zipper/expression/detail/SparseAssignHelper.hpp"
#include "zipper/storage/SparseCoordinateAccessor.hpp"
#include "zipper/storage/detail/SparseCompressedData.hpp"
#include "zipper/storage/detail/to_sparse_compressed_data.hpp"
#include "zipper/storage/layout_types.hpp"
#include "zipper/utils/extents/assignable_extents.hpp"

namespace zipper::storage {

/// @brief Read-only (by default) expression wrapper around
/// SparseCompressedData.
///
/// Provides `coeff()` for all indices (returns 0 for missing entries).
/// When `ValueType` is non-const, also provides `coeff_ref()` and
/// `const_coeff_ref()` for mutable access to *existing* entries (throws
/// for missing entries — use COO format for structural changes).
///
/// Supports `index_set<D>()` for zero-aware expression operations.
///
/// @tparam LayoutPolicy  `layout_right` for CSR (row-compressed, default),
///                       `layout_left` for CSC (column-compressed).
///   - CSR: outer dimension = rows (dim 0), inner dimension = columns (dim 1).
///     Data is sorted by (row, col). `index_set<1>(row)` is O(log R).
///   - CSC: outer dimension = columns (dim 1), inner dimension = rows (dim 0).
///     Data is sorted by (col, row). `index_set<0>(col)` is O(log C).
template <typename ValueType, typename Extents,
          typename LayoutPolicy = default_layout_policy>
class SparseCompressedAccessor
    : public expression::ExpressionBase<
          SparseCompressedAccessor<ValueType, Extents, LayoutPolicy>>,
      public Extents {
public:
  using ParentType = expression::ExpressionBase<
      SparseCompressedAccessor<ValueType, Extents, LayoutPolicy>>;
  using value_type = ValueType;
  using element_type = std::remove_const_t<ValueType>;
  using extents_type = Extents;
  using layout_policy = LayoutPolicy;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  using compressed_data_type =
      detail::SparseCompressedData<element_type, extents_type::rank() - 1>;

  constexpr static auto rank() -> rank_type { return extents_type::rank(); }
  constexpr static bool IsStatic =
      zipper::detail::ExtentsTraits<extents_type>::is_static;

  // CSR (layout_right): outer=0 (rows), inner=1 (cols)
  // CSC (layout_left):  outer=1 (cols), inner=0 (rows)
  // For rank-1, layout is irrelevant — all paths are layout-agnostic.
  constexpr static bool is_csr =
      std::is_same_v<LayoutPolicy, layout_right>;
  constexpr static bool is_csc =
      std::is_same_v<LayoutPolicy, layout_left>;
  static_assert(rank() < 2 || is_csr || is_csc,
      "LayoutPolicy must be layout_right (CSR) or layout_left (CSC) "
      "for rank >= 2");

  // For rank-2: which user-facing dimension is the outer/inner compressed dim.
  // For rank-1: defaulted to 0, but never meaningfully used.
  constexpr static rank_type outer_dim = is_csr ? 0 : (rank() >= 2 ? 1 : 0);
  constexpr static rank_type inner_dim = is_csr ? (rank() >= 2 ? 1 : 0) : 0;

  using extents_type::extent;
  using ParentType::size;
  auto extents() const -> const extents_type & { return *this; }

  // ── Constructors ──────────────────────────────────────────────────────
  ~SparseCompressedAccessor();

  SparseCompressedAccessor()
    requires(IsStatic)
      : extents_type() {}

  SparseCompressedAccessor(const extents_type &extents_)
    requires(!IsStatic)
      : extents_type(extents_) {}

  SparseCompressedAccessor(const SparseCompressedAccessor &) = default;
  SparseCompressedAccessor(SparseCompressedAccessor &&) = default;
  auto operator=(const SparseCompressedAccessor &)
      -> SparseCompressedAccessor & = default;
  auto operator=(SparseCompressedAccessor &&)
      -> SparseCompressedAccessor & = default;

  /// Construct from compressed data + extents.
  SparseCompressedAccessor(compressed_data_type data,
                           const extents_type &extents_)
      : extents_type(extents_), m_data(std::move(data)) {}

  /// Construct from compressed data (static extents only).
  explicit SparseCompressedAccessor(compressed_data_type data)
    requires(IsStatic)
      : extents_type(), m_data(std::move(data)) {}

  /// Construct from a SparseCoordinateAccessor (compresses internally).
  /// Uses ReverseIndices when layout is CSC so that the compressed data
  /// is stored column-major (outer=col, inner=row).
  template <typename VT2>
    requires std::convertible_to<VT2, element_type>
  explicit SparseCompressedAccessor(
      const SparseCoordinateAccessor<VT2, Extents> &coo)
      : extents_type(coo.extents()),
        m_data(detail::to_sparse_compressed_data<LayoutPolicy>(coo)) {}

  // ── Element access ────────────────────────────────────────────────────

  /// Read any element. Returns 0 for non-stored entries.
  /// For CSR: m_data stores (row, col) order — pass indices directly.
  /// For CSC: m_data stores (col, row) order — swap indices.
  template <typename... Indices>
  auto coeff(Indices &&...indices) const -> element_type {
    if constexpr (rank() == 1 || is_csr) {
      return m_data.coeff(static_cast<index_type>(indices)...);
    } else {
      return coeff_swapped(
          std::make_index_sequence<sizeof...(Indices)>{},
          std::forward_as_tuple(std::forward<Indices>(indices)...));
    }
  }

  /// Mutable reference to an *existing* entry. Throws for missing.
  template <typename... Indices>
  auto coeff_ref(Indices &&...indices) -> value_type &
    requires(!std::is_const_v<ValueType>)
  {
    if constexpr (rank() == 1 || is_csr) {
      return m_data.coeff_ref(static_cast<index_type>(indices)...);
    } else {
      return coeff_ref_swapped(
          std::make_index_sequence<sizeof...(Indices)>{},
          std::forward_as_tuple(std::forward<Indices>(indices)...));
    }
  }

  /// Const reference to an *existing* entry. Throws for missing.
  template <typename... Indices>
  auto const_coeff_ref(Indices &&...indices) const -> const element_type &
  {
    if constexpr (rank() == 1 || is_csr) {
      return m_data.const_coeff_ref(static_cast<index_type>(indices)...);
    } else {
      return const_coeff_ref_swapped(
          std::make_index_sequence<sizeof...(Indices)>{},
          std::forward_as_tuple(std::forward<Indices>(indices)...));
    }
  }

  // ── Access to underlying data ─────────────────────────────────────────
  auto compressed_data() const -> const compressed_data_type & {
    return m_data;
  }
  auto compressed_data() -> compressed_data_type &
    requires(!std::is_const_v<ValueType>)
  {
    return m_data;
  }

  /// Clear all stored entries, resetting to an empty compressed container.
  void clear()
    requires(!std::is_const_v<ValueType>)
  {
    m_data = {};
  }

  /// Assign from an arbitrary expression.  Builds an intermediate COO
  /// representation and converts to compressed format.
  ///
  /// Fast path: if the source is a SparseCompressedAccessor with the same
  /// layout, directly copy the compressed data (O(nnz), no allocation).
  template <zipper::concepts::Expression V>
  void assign(const V &v)
    requires(!std::is_const_v<ValueType> &&
             zipper::utils::extents::assignable_extents_v<
                 typename V::extents_type, extents_type>)
  {
    // Fast path: same-layout compressed → compressed copy
    if constexpr (std::is_same_v<std::decay_t<V>,
                                 SparseCompressedAccessor<element_type,
                                                          extents_type,
                                                          LayoutPolicy>>) {
      if constexpr (!IsStatic) {
        static_cast<extents_type&>(*this) = v.extents();
      }
      m_data = v.compressed_data();
      return;
    }
    // Also handle const-qualified source of same layout
    if constexpr (std::is_same_v<std::decay_t<V>,
                                 SparseCompressedAccessor<const element_type,
                                                          extents_type,
                                                          LayoutPolicy>>) {
      if constexpr (!IsStatic) {
        static_cast<extents_type&>(*this) = v.extents();
      }
      m_data = v.compressed_data();
      return;
    }
    SparseCoordinateAccessor<element_type, extents_type> coo;
    if constexpr (!IsStatic) {
      static_cast<extents_type&>(coo) = this->extents();
    }
    expression::detail::SparseAssignHelper::assign(v, coo);
    m_data = detail::to_sparse_compressed_data<LayoutPolicy>(coo);
  }

  // ── index_set: rank-1 ────────────────────────────────────────────────
  template <rank_type D>
    requires(D == 0 && rank() == 1)
  auto index_set() const -> expression::detail::DynamicSparseIndexSet {
    // Base-level m_data stores pairs of (index, value).
    std::vector<index_type> result;
    result.reserve(m_data.size());
    for (const auto &[idx, val] : m_data.m_data) {
      result.push_back(idx);
    }
    return {std::move(result)};
  }

  // ── index_set: rank-2 ────────────────────────────────────────────────
  // For CSR: D==1 (cols for given row) is fast O(log R), D==0 is slow O(nnz).
  // For CSC: D==0 (rows for given col) is fast O(log C), D==1 is slow O(nnz).
  // Generalized: D==inner_dim is fast (spans lookup), D==outer_dim is slow.
  template <rank_type D>
    requires(D < rank() && rank() == 2)
  auto index_set(index_type other_idx) const
      -> expression::detail::DynamicSparseIndexSet {
    if constexpr (D == inner_dim) {
      // Fast path: inner indices for a given outer index.
      // m_spans stores (outer_index, start, size) tuples.
      auto it = std::lower_bound(
          m_data.m_spans.begin(), m_data.m_spans.end(), other_idx,
          [](const std::tuple<index_type, size_t, size_t> &a,
             index_type b) -> auto { return std::get<0>(a) < b; });
      if (it != m_data.m_spans.end() && std::get<0>(*it) == other_idx) {
        auto [outer, start, sz] = *it;
        std::vector<index_type> result;
        result.reserve(sz);
        // Parent (N=0) m_data stores (inner_index, value) pairs.
        using Base = detail::SparseCompressedData<element_type, 0>;
        const auto &base = static_cast<const Base &>(m_data);
        for (size_t i = start; i < start + sz; ++i) {
          result.push_back(base.m_data[i].first);
        }
        return {std::move(result)};
      }
      return {{}};
    } else {
      // Slow path: outer indices for a given inner index.
      // Must scan all spans and check each outer's inner entries.
      using Base = detail::SparseCompressedData<element_type, 0>;
      const auto &base = static_cast<const Base &>(m_data);
      std::vector<index_type> result;
      for (const auto &[outer, start, sz] : m_data.m_spans) {
        auto sp = std::span(base.m_data).subspan(start, sz);
        auto inner_it = std::lower_bound(
            sp.begin(), sp.end(), other_idx,
            [](const std::pair<index_type, element_type> &a,
               index_type b) -> auto { return a.first < b; });
        if (inner_it != sp.end() && inner_it->first == other_idx) {
          result.push_back(outer);
        }
      }
      return {std::move(result)};
    }
  }

  // ── rank-1 convenience ──────────────────────────────────────────────
  auto nonzero_segment() const
    requires(rank() == 1)
  {
    return index_set<0>();
  }

  // ── rank-2 convenience ──────────────────────────────────────────────
  auto col_range_for_row(index_type row) const
    requires(rank() == 2)
  {
    return index_set<1>(row);
  }

  auto row_range_for_col(index_type col) const
    requires(rank() == 2)
  {
    return index_set<0>(col);
  }

private:
  compressed_data_type m_data = {};

  // ── CSC index-swapping helpers (rank-2 only) ────────────────────────
  // For CSC, user passes (row, col) but m_data expects (col, row).
  // We reverse the tuple of indices before forwarding to m_data.
  template <std::size_t... Is, typename Tuple>
  auto coeff_swapped(std::index_sequence<Is...>, Tuple &&t) const
      -> element_type {
    constexpr auto N = sizeof...(Is);
    return m_data.coeff(
        static_cast<index_type>(std::get<N - 1 - Is>(t))...);
  }

  template <std::size_t... Is, typename Tuple>
  auto coeff_ref_swapped(std::index_sequence<Is...>, Tuple &&t)
      -> value_type & {
    constexpr auto N = sizeof...(Is);
    return m_data.coeff_ref(
        static_cast<index_type>(std::get<N - 1 - Is>(t))...);
  }

  template <std::size_t... Is, typename Tuple>
  auto const_coeff_ref_swapped(std::index_sequence<Is...>, Tuple &&t) const
      -> const element_type & {
    constexpr auto N = sizeof...(Is);
    return m_data.const_coeff_ref(
        static_cast<index_type>(std::get<N - 1 - Is>(t))...);
  }
};

template <typename ValueType, typename Extents, typename LayoutPolicy>
SparseCompressedAccessor<ValueType, Extents, LayoutPolicy>::
    ~SparseCompressedAccessor() = default;

} // namespace zipper::storage

namespace zipper::expression {
template <typename ValueType, typename Extents, typename LayoutPolicy>
struct detail::ExpressionTraits<
    zipper::storage::SparseCompressedAccessor<ValueType, Extents, LayoutPolicy>>
    : public detail::BasicExpressionTraits<
          ValueType, Extents,
          zipper::detail::AccessFeatures{
              .is_const = std::is_const_v<ValueType>,
              .is_reference = false},
          zipper::detail::ShapeFeatures{.is_resizable = false}> {
  constexpr static bool has_index_set = true;
  constexpr static bool has_known_zeros = has_index_set;

  /// Sparse compressed leaf → prefer the compressed layout it uses.
  using preferred_layout =
      zipper::detail::SparseLayoutPreference<LayoutPolicy>;
};
} // namespace zipper::expression

#endif
