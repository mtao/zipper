#if !defined(ZIPPER_STORAGE_SPARSECOORDINATEACCESSOR_HPP)
#define ZIPPER_STORAGE_SPARSECOORDINATEACCESSOR_HPP

#include <algorithm>
#include <format>
#include <ranges>
#include <sstream>

#include "zipper/detail/assert.hpp"
#include <vector>

#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/expression/ExpressionBase.hpp"
#include "zipper/expression/detail/IndexSet.hpp"
#include "zipper/expression/detail/SparseAssignHelper.hpp"
#include "zipper/utils/extents/assignable_extents.hpp"
namespace zipper::storage {
template <typename ValueType, typename Extents> class SparseCoordinateAccessor;

namespace detail {

template <typename ValueType, typename Extents, bool is_const>
class SparseCoordinateAccessor_iterator {
public:
  using value_type = SparseCoordinateAccessor_iterator;
  using difference_type = int64_t;
  using pointer = void;
  using reference = value_type &;
  using iterator_category = std::random_access_iterator_tag;

  SparseCoordinateAccessor_iterator(
      const SparseCoordinateAccessor<ValueType, Extents> &a, size_t i)
      : m_acc(a), m_index(i) {}
  template <bool ic>
  SparseCoordinateAccessor_iterator(
      const SparseCoordinateAccessor_iterator<ValueType, Extents, ic> &o)
      : SparseCoordinateAccessor_iterator(o.m_acc, o.m_index) {}
  SparseCoordinateAccessor_iterator(
      const SparseCoordinateAccessor_iterator &o) = default;
  SparseCoordinateAccessor_iterator(SparseCoordinateAccessor_iterator &&o) =
      default;

  [[nodiscard]] auto index() const -> size_t { return m_index; }
  auto multiindex() const { return m_acc.get_multiindex(index()); }
  auto value() const -> ValueType { return m_acc.m_data.at(index()); }
  auto operator=(const SparseCoordinateAccessor_iterator &N)
      -> SparseCoordinateAccessor_iterator & {
    ZIPPER_ASSERT(&m_acc == &N.m_acc);
    m_index = N.m_index;
    return *this;
  }
  auto operator+=(size_t N) -> SparseCoordinateAccessor_iterator & {
    m_index += N;
    return *this;
  }
  auto operator++() -> SparseCoordinateAccessor_iterator & {
    ++m_index;
    return *this;
  }
  auto operator--() -> SparseCoordinateAccessor_iterator & {
    --m_index;
    return *this;
  }
  auto operator-(const SparseCoordinateAccessor_iterator &o) const -> int64_t {
    return int64_t(m_index) - o.m_index;
  }
  auto operator*() const -> const auto & { return *this; }
  auto operator*() -> auto & { return *this; }
  /// Iterator-to-iterator comparison (by storage index). Works in both
  /// compressed and uncompressed states — used by range-for in find().
  auto operator<=>(const SparseCoordinateAccessor_iterator &t) const
      -> std::strong_ordering {
    return index() <=> t.index();
  }
  auto operator==(const SparseCoordinateAccessor_iterator &o) const -> bool {
    return index() == o.index();
  }

  /// Iterator-to-multiindex comparison (lexicographic). Only valid when
  /// the underlying data is compressed and sorted.
  template <typename T>
    requires(!std::is_same_v<std::decay_t<T>,
                             SparseCoordinateAccessor_iterator>)
  auto operator<=>(const T &t) const -> std::strong_ordering {
    ZIPPER_ASSERT(m_acc.m_compressed);
    ZIPPER_ASSERT(index() != m_acc.size());
    return multiindex() <=> t;
  }
  template <typename T>
    requires(!std::is_same_v<std::decay_t<T>,
                             SparseCoordinateAccessor_iterator>)
  auto operator==(const T &o) const -> bool {
    return std::is_eq(*this <=> o);
  }
  auto operator<(const auto &o) const -> bool {
    return std::is_lt(*this <=> o);
  }

private:
  const SparseCoordinateAccessor<ValueType, Extents> &m_acc;
  size_t m_index;
};

} // namespace detail
// SpanStorage predeclares the defaults now?
template <typename ValueType, typename Extents>
class SparseCoordinateAccessor
    : public expression::ExpressionBase<
          SparseCoordinateAccessor<ValueType, Extents>>,
      public Extents {

public:
  template <typename VT, typename E, bool is_const>
  friend class detail::SparseCoordinateAccessor_iterator;
  using ParentType = expression::ExpressionBase<
      SparseCoordinateAccessor<ValueType, Extents>>;
  using value_type = ValueType;
  using element_type = std::remove_const_t<ValueType>;
  using extents_type = Extents;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  constexpr static auto rank() -> rank_type { return extents_type::rank(); }
  constexpr static bool IsStatic =
      zipper::detail::ExtentsTraits<extents_type>::is_static;

  using extents_type::extent;
  using ParentType::size;
  auto extents() const -> const extents_type & { return *this; }

  ~SparseCoordinateAccessor();
  SparseCoordinateAccessor()
    requires(IsStatic)
      : extents_type() {}

  SparseCoordinateAccessor(const SparseCoordinateAccessor &) = default;
  SparseCoordinateAccessor(SparseCoordinateAccessor &&) = default;
  auto operator=(const SparseCoordinateAccessor &)
      -> SparseCoordinateAccessor & = default;
  auto operator=(SparseCoordinateAccessor &&)
      -> SparseCoordinateAccessor & = default;

  SparseCoordinateAccessor(const extents_type &extents_)
    requires(!IsStatic)
      : extents_type(extents_) {}

  [[nodiscard]] auto is_compressed() const -> bool { return m_compressed; }

private:
  [[nodiscard]] auto data_size() const -> size_t { return m_data.size(); }

  template <size_t... N>
  auto get_multiindex(std::integer_sequence<size_t, N...>, size_t a) const {
    return std::tie(m_indices[N][a]...);
  }
  auto get_multiindex(size_t a) const {
    return get_multiindex(std::make_integer_sequence<size_t, rank()>{}, a);
  }

  template <typename... Indices>
  auto index_lexicographic_compare(size_t a, Indices &&...indices) const
      -> std::strong_ordering {
    return get_multiindex(a) <=> std::tie(indices...);
  }

  [[nodiscard]] auto index_lexicographic_compare(size_t a, size_t b) const
      -> std::strong_ordering {
    return get_multiindex(a) <=> get_multiindex(b);
  }
  template <rank_type... N, typename... Indices>
    requires(sizeof...(Indices) == rank())
  auto emplace(std::integer_sequence<rank_type, N...>, Indices... indices)
      -> value_type & {
    auto add = [&]<rank_type J>(std::integral_constant<rank_type, J>) -> auto {
      m_indices[J].emplace_back(zipper::detail::pack_index<J>(indices...));
    };
    (add(std::integral_constant<rank_type, N>{}), ...);
    m_compressed = false;
    return m_data.emplace_back(0);
  }

public:
  using iterator_type =
      detail::SparseCoordinateAccessor_iterator<ValueType, Extents, false>;
  using const_iterator_type =
      detail::SparseCoordinateAccessor_iterator<ValueType, Extents, true>;
  auto begin() -> iterator_type { return iterator_type{*this, 0}; }
  auto end() -> iterator_type { return iterator_type{*this, data_size()}; }
  auto begin() const -> const_iterator_type {
    return const_iterator_type{*this, 0};
  }
  auto end() const -> const_iterator_type {
    return const_iterator_type{*this, data_size()};
  }
  auto cbegin() const -> const_iterator_type {
    return const_iterator_type{*this, 0};
  }
  auto cend() const -> const_iterator_type {
    return const_iterator_type{*this, data_size()};
  }

  /// Remove all stored entries, resetting to an empty sparse container.
  void clear() {
    m_data.clear();
    for (auto &idx_vec : m_indices) {
      idx_vec.clear();
    }
    m_compressed = true;
  }

  /// Reserve capacity for at least `n` entries to avoid reallocations
  /// during bulk emplacement.
  void reserve(size_t n) {
    m_data.reserve(n);
    for (auto &idx_vec : m_indices) {
      idx_vec.reserve(n);
    }
  }

  /// Assign from an arbitrary expression.  Clears existing entries and
  /// emplaces only nonzero values from the source.
  template <zipper::concepts::Expression V>
  void assign(const V &v)
    requires(!std::is_const_v<ValueType> &&
             zipper::utils::extents::assignable_extents_v<
                typename V::extents_type, extents_type>)
  {
    expression::detail::SparseAssignHelper::assign(v, *this);
  }

  void compress() {
    if (data_size() == 0) {
      m_compressed = true;
      return;
    }
    std::vector o = std::views::iota(size_t(0), data_size()) |
                    std::ranges::to<std::vector>();

    std::ranges::sort(o, [this](size_t a, size_t b) -> auto {
      return std::is_lt(index_lexicographic_compare(a, b));
    });

    std::vector<size_t> o2;
    o2.emplace_back(o[0]);
    for (size_t j = 1; j < data_size(); ++j) {
      size_t cur = o2.back();
      if (get_multiindex(cur) == get_multiindex(o[j])) {
        m_data[cur] += m_data[o[j]];
      } else {
        o2.emplace_back(o[j]);
      }
    }
    o = std::move(o2);

    auto reorder = [&o]<typename T>(std::vector<T> &data) -> auto {
      data = std::views::transform(
                 o, [&data](size_t j) noexcept -> auto { return data[j]; }) |
             std::ranges::to<std::vector<T>>();
    };
    reorder(m_data);
    for (auto &d : m_indices) {
      reorder(d);
    }

    m_compressed = true;
  }
  template <concepts::Index... Indices>
  auto find(Indices &&...indices) const -> const_iterator_type {
    ZIPPER_ASSERT(zipper::utils::extents::indices_in_range(extents(), indices...));
    if (m_compressed) {
      if (data_size() == 0) {
        return end();
      }
      auto it = std::lower_bound(begin(), end(), std::tie(indices...),
                                 [](const auto &a, const auto &b) -> auto {
                                   return a.multiindex() < b;
                                 });
      if (it != end() && it.multiindex() == std::tie(indices...)) {
        return it;
      }
    } else {
      for (const auto &it : *this) {
        if (std::tie(indices...) == it.multiindex()) {
          return it;
        }
      }
    }
    return end();
  }
  template <typename... Indices>
  auto get_index(Indices &&...indices) const -> index_type {
    return find(indices...).index();
  }

public:
  template <typename... Indices>
  auto coeff(Indices &&...indices) const -> value_type {
    index_type idx = get_index(std::forward<Indices>(indices)...);
    if (idx != data_size()) {
      return m_data.at(idx);
    } else {
      return 0;
    }
  }
  template <typename... Indices>
  auto coeff_ref(Indices &&...indices) -> value_type &
    requires(!std::is_const_v<ValueType>)
  {
    index_type idx = get_index(std::forward<Indices>(indices)...);
    if (idx != data_size()) {
      return m_data.at(idx);
    } else {
      std::ostringstream oss;
      ((oss << indices << ' '), ...);
      throw std::invalid_argument(
          "Index (" + oss.str() +
          ") did not have a value in SparseCoordinateAccessor");
    }
  }

  template <typename... Indices>
  auto const_coeff_ref(Indices &&...indices) const -> const value_type & {
    index_type idx = get_index(std::forward<Indices>(indices)...);
    if (idx != data_size()) {
      return m_data.at(idx);
    } else {
      std::ostringstream oss;
      ((oss << indices << ' '), ...);
      throw std::invalid_argument(
          "Index (" + oss.str() +
          ") did not have a value in SparseCoordinateAccessor");
    }
  }
  template <typename... Indices>
    requires(sizeof...(Indices) == rank())
  auto emplace(Indices... indices) -> value_type & {
    return emplace(std::make_integer_sequence<rank_type, rank()>{}, indices...);
  }

  // ── index_set: rank-1 (no argument, returns all stored indices) ──────────
  template <rank_type D>
    requires(D == 0 && rank() == 1)
  auto index_set() const -> expression::detail::SpanSparseIndexSet {
    ZIPPER_ASSERT(m_compressed);
    return {std::span<const index_type>(m_indices[0])};
  }

  // ── index_set: rank-2 (returns indices along dimension D for other_idx) ──
  template <rank_type D>
    requires(D < rank() && rank() == 2)
  auto index_set(index_type other_idx) const {
    ZIPPER_ASSERT(m_compressed);
    if constexpr (D == 1) {
      // Column indices for a given row — contiguous after compression
      auto lo = std::lower_bound(m_indices[0].begin(), m_indices[0].end(),
                                 other_idx);
      auto hi = std::upper_bound(lo, m_indices[0].end(), other_idx);
      auto lo_off =
          static_cast<size_t>(std::distance(m_indices[0].begin(), lo));
      auto count = static_cast<size_t>(std::distance(lo, hi));
      return expression::detail::SpanSparseIndexSet{
          std::span<const index_type>(m_indices[1].data() + lo_off, count)};
    } else {
      // Row indices for a given column — must scan all entries
      std::vector<index_type> result;
      for (size_t i = 0; i < data_size(); ++i) {
        if (m_indices[1][i] == other_idx) {
          result.push_back(m_indices[0][i]);
        }
      }
      return expression::detail::DynamicSparseIndexSet{std::move(result)};
    }
  }

  // ── rank-1 convenience ───────────────────────────────────────────────────
  auto nonzero_segment() const
    requires(rank() == 1)
  {
    return index_set<0>();
  }

  // ── rank-2 convenience ───────────────────────────────────────────────────
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
  std::vector<value_type> m_data = {};
  std::array<std::vector<index_type>, rank()> m_indices = {};
  bool m_compressed = true;
};
template <typename ValueType, typename Extents>
SparseCoordinateAccessor<ValueType, Extents>::~SparseCoordinateAccessor() =
    default;
} // namespace zipper::storage

namespace zipper::expression {
template <typename ValueType, typename Extents>
struct detail::ExpressionTraits<
    zipper::storage::SparseCoordinateAccessor<ValueType, Extents>>
    : public detail::BasicExpressionTraits<
          ValueType, Extents,
          zipper::detail::AccessFeatures{std::is_const_v<ValueType>, false},
          zipper::detail::ShapeFeatures::fixed()> {
  constexpr static bool has_index_set = true;
  constexpr static bool has_known_zeros = has_index_set;

  /// COO has no compressed layout bias → no preference.
  using preferred_layout = zipper::detail::NoLayoutPreference;
};

} // namespace zipper::expression

#endif
