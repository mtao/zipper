#if !defined(ZIPPER_STORAGE_SPARSECOMPRESSEDACCESSOR_HPP)
#define ZIPPER_STORAGE_SPARSECOMPRESSEDACCESSOR_HPP

#include <stdexcept>
#include <vector>

#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/expression/ExpressionBase.hpp"
#include "zipper/expression/detail/IndexSet.hpp"
#include "zipper/storage/SparseCoordinateAccessor.hpp"
#include "zipper/storage/detail/SparseCompressedData.hpp"
#include "zipper/storage/detail/to_sparse_compressed_data.hpp"

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
template <typename ValueType, typename Extents>
class SparseCompressedAccessor
    : public expression::ExpressionBase<
          SparseCompressedAccessor<ValueType, Extents>>,
      public Extents {
public:
  using ParentType = expression::ExpressionBase<
      SparseCompressedAccessor<ValueType, Extents>>;
  using value_type = ValueType;
  using element_type = std::remove_const_t<ValueType>;
  using extents_type = Extents;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  using compressed_data_type =
      detail::SparseCompressedData<element_type, extents_type::rank() - 1>;

  constexpr static auto rank() -> rank_type { return extents_type::rank(); }
  constexpr static bool IsStatic =
      zipper::detail::ExtentsTraits<extents_type>::is_static;

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
  template <typename VT2>
    requires std::convertible_to<VT2, element_type>
  explicit SparseCompressedAccessor(
      const SparseCoordinateAccessor<VT2, Extents> &coo)
      : extents_type(coo.extents()),
        m_data(detail::to_sparse_compressed_data(coo)) {}

  // ── Element access ────────────────────────────────────────────────────

  /// Read any element. Returns 0 for non-stored entries.
  template <typename... Indices>
  auto coeff(Indices &&...indices) const -> element_type {
    return m_data.coeff(static_cast<index_type>(indices)...);
  }

  /// Mutable reference to an *existing* entry. Throws for missing.
  template <typename... Indices>
  auto coeff_ref(Indices &&...indices) -> value_type &
    requires(!std::is_const_v<ValueType>)
  {
    return m_data.coeff_ref(static_cast<index_type>(indices)...);
  }

  /// Const reference to an *existing* entry. Throws for missing.
  template <typename... Indices>
  auto const_coeff_ref(Indices &&...indices) const -> const element_type &
    requires(!std::is_const_v<ValueType>)
  {
    return m_data.const_coeff_ref(static_cast<index_type>(indices)...);
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
  template <rank_type D>
    requires(D < rank() && rank() == 2)
  auto index_set(index_type other_idx) const
      -> expression::detail::DynamicSparseIndexSet {
    if constexpr (D == 1) {
      // Column indices for a given row.
      // m_spans stores (row_index, start, size) tuples.
      auto it = std::lower_bound(
          m_data.m_spans.begin(), m_data.m_spans.end(), other_idx,
          [](const std::tuple<index_type, size_t, size_t> &a,
             index_type b) -> auto { return std::get<0>(a) < b; });
      if (it != m_data.m_spans.end() && std::get<0>(*it) == other_idx) {
        auto [row, start, sz] = *it;
        std::vector<index_type> result;
        result.reserve(sz);
        // Parent (N=0) m_data stores (col_index, value) pairs.
        using Base = detail::SparseCompressedData<element_type, 0>;
        const auto &base = static_cast<const Base &>(m_data);
        for (size_t i = start; i < start + sz; ++i) {
          result.push_back(base.m_data[i].first);
        }
        return {std::move(result)};
      }
      return {{}};
    } else {
      // Row indices for a given column.
      // Must scan all spans and check each row's columns.
      using Base = detail::SparseCompressedData<element_type, 0>;
      const auto &base = static_cast<const Base &>(m_data);
      std::vector<index_type> result;
      for (const auto &[row, start, sz] : m_data.m_spans) {
        auto sp = std::span(base.m_data).subspan(start, sz);
        auto col_it = std::lower_bound(
            sp.begin(), sp.end(), other_idx,
            [](const std::pair<index_type, element_type> &a,
               index_type b) -> auto { return a.first < b; });
        if (col_it != sp.end() && col_it->first == other_idx) {
          result.push_back(row);
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
};

template <typename ValueType, typename Extents>
SparseCompressedAccessor<ValueType, Extents>::~SparseCompressedAccessor() =
    default;

} // namespace zipper::storage

namespace zipper::expression {
template <typename ValueType, typename Extents>
struct detail::ExpressionTraits<
    zipper::storage::SparseCompressedAccessor<ValueType, Extents>>
    : public detail::BasicExpressionTraits<
          ValueType, Extents,
          zipper::detail::AccessFeatures{
              .is_const = std::is_const_v<ValueType>,
              .is_reference = false},
          zipper::detail::ShapeFeatures{.is_resizable = false}> {
  constexpr static bool has_index_set = true;
  constexpr static bool has_known_zeros = has_index_set;
};
} // namespace zipper::expression

#endif
