#include <algorithm>
#if !defined(ZIPPER_STORAGE_SPARSEACCESSOR_HPP)
#define ZIPPER_STORAGE_SPARSEACCESSOR_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-compare"
#pragma GCC diagnostic ignored "-Winline"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wswitch-enum"
#pragma GCC diagnostic ignored "-Wswitch-default"
#pragma GCC diagnostic ignored "-Wstrict-overflow"
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
#pragma GCC diagnostic ignored "-Weffc++"
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmultiple-inheritance"
#pragma GCC diagnostic ignored "-Wabi-tag"
#endif
#include <fmt/format.h>
#include <fmt/ranges.h>
#pragma GCC diagnostic pop

#include <ranges>
#include <vector>

#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/expression/SizedExpressionBase.hpp"
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
  /*
ValueType& value()
  requires(!is_const)
{
  return m_acc.m_data.at(index());
}
*/
  auto operator=(const SparseCoordinateAccessor_iterator &N)
      -> SparseCoordinateAccessor_iterator & {
    assert(&m_acc == &N.m_acc);
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
  auto operator<=>(const auto &t) const -> std::strong_ordering {
    assert(m_acc.m_compressed);
    if constexpr (std::is_same_v<SparseCoordinateAccessor_iterator<
                                     ValueType, Extents, is_const>,
                                 std::decay_t<decltype(t)>>) {
      return index() <=> t.index();
      // return multiindex() <=> t.multiindex();
    } else {
      assert(index() != m_acc.size());
      return multiindex() <=> t;
    }
  }
  auto operator==(const auto &o) const -> bool {
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
    : public expression::SizedExpressionBase<
          SparseCoordinateAccessor<ValueType, Extents>> {
  // template <typename T>
  // struct _detail;
  // template <rank_type... Index>
  // struct _detail<std::integer_sequence<rank_type, Index...>> {
  //     // storage_type
  // };

  // using detail =
  //     _detail<std::make_integer_sequence<rank_type, Extents::rank()>>;

public:
  template <typename VT, typename E, bool is_const>
  friend class detail::SparseCoordinateAccessor_iterator;
  using ParentType = expression::SizedExpressionBase<
      SparseCoordinateAccessor<ValueType, Extents>>;
  using value_type = ValueType;
  using extents_type = Extents;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  constexpr static auto rank() -> rank_type { return extents_type::rank(); }
  constexpr static bool IsStatic =
      zipper::detail::ExtentsTraits<extents_type>::is_static;

  using ParentType::extents;
  using ParentType::size;

  ~SparseCoordinateAccessor();
  SparseCoordinateAccessor()
    requires(IsStatic)
      : ParentType() {}

  SparseCoordinateAccessor(const SparseCoordinateAccessor &) = default;
  SparseCoordinateAccessor(SparseCoordinateAccessor &&) = default;
  auto operator=(const SparseCoordinateAccessor &)
      -> SparseCoordinateAccessor & = default;
  auto operator=(SparseCoordinateAccessor &&)
      -> SparseCoordinateAccessor & = default;

  SparseCoordinateAccessor(const extents_type &extents_)
    requires(!IsStatic)
      : ParentType(extents_) {}

  // template <zipper::concepts::ExtentsType E2>
  // void resize(const E2& e)
  //     requires(extents_traits::template is_convertable_from<E2>() &&
  //              !IsStatic)
  //{
  //     static_assert(E2::rank() != 0);
  //     this->resize_extents(e);
  // }

  [[nodiscard]] auto is_compressed() const -> bool { return m_compressed; }

private:
  [[nodiscard]] auto data_size() const -> size_t { return m_data.size(); }

  template <size_t... N>
  auto get_multiindex(std::integer_sequence<size_t, N...>, size_t a) const {
    return std::tie(m_indices[N][a]...);
  }
  template <size_t... N> auto get_multiindex(size_t a) const {
    return get_multiindex(std::make_integer_sequence<size_t, rank()>{}, a);
  }

  template <typename... Indices>
  auto index_lexicographic_compare(size_t a, Indices &&...indices) const
      -> std::strong_ordering {
    return get_multiindex(a) <=> std::tie(indices...);
    /*
    for(size_t j = 0; j < rank(); ++j) {
        const index_type ai = m_indices[j][a];
        const index_type bi = m_indices[j][b];
        auto cmp = ai <=> bi;
        if(cmp == 0) {
            continue;
        } else {
            return cmp;
        }
    }
    return std::strong_ordering::equal;
    */
  }

  [[nodiscard]] auto index_lexicographic_compare(size_t a, size_t b) const
      -> std::strong_ordering {
    return get_multiindex(a) <=> get_multiindex(b);
    /*
    for(size_t j = 0; j < rank(); ++j) {
        const index_type ai = m_indices[j][a];
        const index_type bi = m_indices[j][b];
        auto cmp = ai <=> bi;
        if(cmp == 0) {
            continue;
        } else {
            return cmp;
        }
    }
    return std::strong_ordering::equal;
    */
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

  void compress() {
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
#if !defined(NDEBUG)
    assert(zipper::utils::extents::indices_in_range(extents(), indices...));
#endif
    if (m_compressed) {
      if (data_size() == 0) {
        return end();
      }
      auto it = std::lower_bound(begin(), end(), std::tie(indices...),
                                 [](const auto &a, const auto &b) -> auto {
                                   return a.multiindex() < b;
                                 });
      if (it.multiindex() == std::tie(indices...)) {
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
  auto coeff_ref(Indices &&...indices) -> value_type & {
    index_type idx = get_index(std::forward<Indices>(indices)...);
    if (idx != data_size()) {
      return m_data.at(idx);
    } else {
      throw std::invalid_argument(fmt::format(
          "Index {} did not have a value in SparseCoordinateStorage",
          fmt::join(std::tie(indices...), ",")));
    }
  }

  template <typename... Indices>
  auto const_coeff_ref(Indices &&...indices) const -> const value_type & {
    index_type idx = get_index(std::forward<Indices>(indices)...);
    if (idx != data_size()) {
      return m_data.at(idx);
    } else {
      throw std::invalid_argument(fmt::format(
          "Index {} did not have a value in SparseCoordinateStorage",
          std::tie(indices...)));
    }
  }
  template <typename... Indices>
    requires(sizeof...(Indices) == rank())
  auto emplace(Indices... indices) -> value_type & {
    return emplace(std::make_integer_sequence<rank_type, rank()>{}, indices...);
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
    : public detail::DefaultExpressionTraits<ValueType, Extents>
/*: public detail::ExpressionTraits <
  expression::StorageExpressionBase<zipper::storage::SparseCoordinateStorage<
      ValueType, Extents, LayoutPolicy, AccessorPolicy>> */
{
  using value_type = ValueType;
  using extents_type = Extents;
  constexpr static bool is_writable = true;
  constexpr static bool is_coefficient_consistent = true;
};

} // namespace zipper::expression

namespace std {

template <typename ValueType, typename Extents, bool is_const>
auto distance(const zipper::storage::detail::SparseCoordinateAccessor_iterator<
                  ValueType, Extents, is_const> &a,
              const zipper::storage::detail::SparseCoordinateAccessor_iterator<
                  ValueType, Extents, is_const> &b) -> std::int64_t {
  return std::int64_t(b.index()) - std::int64_t(a.index());
}
template <typename ValueType, typename Extents, bool is_const>
void advance(zipper::storage::detail::SparseCoordinateAccessor_iterator<
                 ValueType, Extents, is_const> &a,
             int64_t n) {
  a += n;
}
} // namespace std
#endif
