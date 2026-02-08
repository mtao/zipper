#if !defined(ZIPPER_STORAGE_DETAIL_STORAGE_TRAITS_HPP)
#define ZIPPER_STORAGE_DETAIL_STORAGE_TRAITS_HPP

#include "extents/dynamic_extents_indices.hpp"
#include "zipper/types.hpp"
#include "zipper/utils/extents/convert_extents.hpp"

namespace zipper::detail {

template <typename> struct extent_values;
template <index_type... Idxs> struct extent_values<zipper::extents<Idxs...>> {
  constexpr static index_type static_sub_size =
      ((Idxs == std::dynamic_extent ? 1 : Idxs) * ... * 1);
};

template <concepts::Extents Extents> struct ExtentsTraits {
  using extents_type = Extents;
  constexpr static rank_type rank = extents_type::rank();
  constexpr static rank_type rank_dynamic = extents_type::rank_dynamic();
  constexpr static rank_type rank_static = rank - rank_dynamic;
  constexpr static bool is_dynamic = rank_dynamic != 0;
  constexpr static bool is_static = !is_dynamic;

  constexpr static index_type static_sub_size =
      extent_values<extents_type>::static_sub_size;

  constexpr static index_type static_size =
      is_dynamic ? std::dynamic_extent : static_sub_size;

  template <typename T>
  using span_type = std::conditional_t<is_static, std::span<T, static_size>,
                                       std::span<T, std::dynamic_extent>>;

  static constexpr auto size(const extents_type &e) -> index_type {
    if constexpr (is_static) {
      return static_size;
    } else {
      index_type s = 1;
      for (rank_type j = 0; j < e.rank(); ++j) {
        s *= e.extent(j);
      }
      return s;
    }
  }

  consteval static auto is_dynamic_extent(rank_type i) -> bool {
    return extents_type::static_extent(i) == std::dynamic_extent;
  }

  template <index_type... Indices>
  constexpr static auto convert_from(const zipper::extents<Indices...> &o)
      -> extents_type {
    if constexpr (sizeof...(Indices) == 0 && is_static) {
      return {};
    } else {
      return utils::extents::convert_extents<Extents>(o);
    }
  }

  template <concepts::Extents Ext>
  constexpr static auto is_convertable_from() -> bool {
    return utils::extents::assignable_extents_v<Ext, extents_type>;
  }
  template <concepts::Extents Ext>
  constexpr static auto is_convertable_from(const Ext &o) -> bool {
    return utils::extents::assignable_extents<Ext, extents_type>::value_runtime(
        o);
  }

  template <concepts::Extents Ext>
  constexpr static auto is_resizeable_from() -> bool {
    return (Ext::rank() == 0 && is_static) || is_convertable_from<Ext>();
  }
  template <concepts::Extents Ext>
  constexpr static auto is_resizeable_from(const Ext &o) -> bool {
    return (Ext::rank() == 0 && is_static) || is_convertable_from(o);
  }

  using dynamic_indices_helper = extents::DynamicExtentIndices<extents_type>;

  constexpr static std::array<rank_type, rank_dynamic> dynamic_indices =
      dynamic_indices_helper::value;

  /// Constructs an extents_type from any object that provides
  /// extent(rank_type) -> index_type.  Only the dynamic extents are
  /// forwarded to the constructor; static extents are known at compile time.
  template <typename HasExtent>
  static constexpr auto make_extents_from(const HasExtent &src)
      -> extents_type {
    if constexpr (is_static) {
      return extents_type{};
    } else {
      return [&]<std::size_t... N>(std::index_sequence<N...>) -> auto {
        return extents_type{src.extent(dynamic_indices[N])...};
      }(std::make_index_sequence<rank_dynamic>{});
    }
  }
};
} // namespace zipper::detail
#endif
