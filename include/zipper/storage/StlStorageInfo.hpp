#if !defined(ZIPPER_STORAGE_STLStorage_HPP)
#define ZIPPER_STORAGE_STLStorage_HPP
#include <array>
#include <cassert>
#include <vector>

#include "zipper/concepts/Index.hpp"
#include "zipper/detail/extents/dynamic_extents_indices.hpp"
#include "zipper/types.hpp"

namespace zipper::storage {

namespace detail {
// internal helper for managing storage info, order of its extents are indexed
// in reverse to the final extents used because the left-most (0th extent) is
// the outermost structure
template <typename T> struct StlStorageInfo {
  using value_type = T;
  constexpr static rank_type rank = 0;
  static auto get_value(T &A) -> auto & { return A; }
  static auto get_value(const T &A) -> const auto & { return A; }
  template <rank_type N> using get_type = std::enable_if<(N == -1), value_type>;

  consteval auto is_valid_extent(const T &) -> bool { return true; }

  static auto initialize(const value_type &t) -> value_type { return t; }
  static void resize(value_type &) {}
};
template <typename S>
struct StlStorageInfo<std::vector<S>> : public StlStorageInfo<S> {
  using self_type = std::vector<S>;
  constexpr static rank_type rank = StlStorageInfo<S>::rank + 1;
  using value_type = typename StlStorageInfo<S>::value_type;
  constexpr static index_type my_static_extent = std::dynamic_extent;
  template <typename... Args>
  static auto get_value(std::vector<S> &A, index_type i, Args &&...args)
      -> auto & {
    return StlStorageInfo<S>::get_value(A[std::size_t{i}],
                                        std::forward<Args>(args)...);
  }
  template <typename... Args>
  static auto get_value(const std::vector<S> &A, index_type i, Args &&...args)
      -> const auto & {
    return StlStorageInfo<S>::get_value(A[std::size_t{i}],
                                        std::forward<Args>(args)...);
  }

  template <rank_type N>
    requires(rank >= 0)
  using get_type =
      std::conditional_t<N + 1 == rank, self_type,
                         typename StlStorageInfo<S>::template get_type<N>>;

  template <rank_type N> consteval static auto static_extent() -> index_type {
    if constexpr (N + 1 == rank) {
      return my_static_extent;
    } else {
      return StlStorageInfo<get_type<N>>::template static_extent<N>();
    }
  }
  template <rank_type M> static auto extent(const self_type &s) -> index_type {
    if constexpr (M + 1 == rank) {
      return s.size();
    } else if constexpr (s.size() == 0) {
      return 0;
    } else {
      return StlStorageInfo<S>::template extent<M>(s[0]);
    }
  }
  template <typename... Args>
  static consteval auto is_valid_extent(const self_type &v, index_type size,
                                        Args &&...args) -> bool {
    if (v.size() != size) {
      return false;
    }
    for (const auto &x : v) {
      if (!StlStorageInfo<S>::is_valid_extent(x, std::forward<Args>(args)...)) {
        return false;
      }
    }
    return true;
  }
  template <typename... Args>
  static auto initialize(const value_type &d, index_type size, Args &&...args)
      -> self_type {
    self_type v(size, StlStorageInfo<S>::initialize(d, args...));
  }

  template <typename... Args>
  static void resize(self_type &s, index_type size, Args &&...args) {
    s.resize(size);
    for (auto &x : s) {
      StlStorageInfo<S>::resize(x, args...);
    }
  }
};
template <size_t N, typename S>
struct StlStorageInfo<std::array<S, N>> : public StlStorageInfo<S> {
  using self_type = std::array<S, N>;
  constexpr static rank_type rank = StlStorageInfo<S>::rank + 1;
  constexpr static index_type my_static_extent = index_type(N);
  using value_type = typename StlStorageInfo<S>::value_type;
  template <typename... Args>
  static auto get_value(std::array<S, N> &A, index_type i, Args &&...args)
      -> auto & {
    return StlStorageInfo<S>::get_value(A[std::size_t{i}],
                                        std::forward<Args>(args)...);
  }
  template <typename... Args>
  static auto get_value(const std::array<S, N> &A, index_type i, Args &&...args)
      -> const auto & {
    return StlStorageInfo<S>::get_value(A[std::size_t{i}],
                                        std::forward<Args>(args)...);
  }
  template <rank_type M>
    requires(rank >= 0)
  using get_type =
      std::conditional_t<M + 1 == rank, self_type,
                         typename StlStorageInfo<S>::template get_type<M>>;

  template <rank_type M> consteval static auto static_extent() -> index_type {
    if constexpr (M + 1 == rank) {
      return my_static_extent;
    } else {
      return StlStorageInfo<get_type<M>>::template static_extent<M>();
    }
  }
  template <rank_type M>
  constexpr static auto extent(const self_type &s) -> index_type {
    if constexpr (M + 1 == rank) {
      return my_static_extent;
      ;
    } else if constexpr (N == 0) {
      return 0;
    } else {
      return StlStorageInfo<S>::template extent<M>(s[0]);
    }
  }
  template <typename> struct detail_;
  template <rank_type... Ms>
  struct detail_<std::integer_sequence<rank_type, Ms...>> {
    //
  };
  using detail =
      detail_<decltype(std::make_integer_sequence<rank_type, rank>{})>;

  template <typename... Args>
  static consteval auto is_valid_extent(const self_type &v, index_type size,
                                        Args &&...args) -> bool {
    if (v.size() != size) {
      return false;
    }
    for (const auto &x : v) {
      if (!StlStorageInfo<S>::is_valid_extent(x, std::forward<Args>(args)...)) {
        return false;
      }
    }
    return true;
  }
  template <typename... Args>
  static auto initialize(const value_type &d, index_type size, Args &&...args)
      -> self_type {
    assert(size == my_static_extent);
    self_type v;
    for (auto &x : v) {
      x = StlStorageInfo<S>::initialize(d, args...);
    }
    return v;
  }
  static auto initialize(const value_type &d) -> self_type {
    self_type v;
    for (auto &x : v) {
      x = StlStorageInfo<S>::initialize(d);
    }
    return v;
  }

  template <typename... Args>
  static void resize(self_type &s, index_type size, Args &&...args) {
    assert(size == my_static_extent);
    for (auto &x : s) {
      StlStorageInfo<S>::resize(x, args...);
    }
  }

  // using extents_type = zipper::extents<N>();
  // constexpr static extents_type extents(const std::array<int64_t, N>& A) {
  //     return extents_type();
  // }
};
// using extents_type =

} // namespace detail

template <typename S> struct StlStorageInfo : public detail::StlStorageInfo<S> {
  using internal_info = detail::StlStorageInfo<S>;

  constexpr static rank_type rank = internal_info::rank;
  using value_type = typename internal_info::value_type;
  template <typename> struct detail_;
  template <rank_type... Ms>
  struct detail_<std::integer_sequence<rank_type, Ms...>> {
    using extents_type =
        extents<internal_info::template static_extent<rank - 1 - Ms>()...>;
    //
    static auto initialize(const extents_type &e,
                           const value_type &default_value) -> S {
      return internal_info::initialize(default_value, e.extent(Ms)...);
    }
    static void resize(S &s, const extents_type &e) {
      internal_info::resize(s, e.extent(Ms)...);
    }
  };
  using detail =
      detail_<decltype(std::make_integer_sequence<rank_type, rank>{})>;
  using extents_type = typename detail::extents_type;

  template <rank_type... Ms>
  static auto make_extents(const S &s, std::integer_sequence<rank_type, Ms...>)
      -> extents_type {
    constexpr auto dyn_indices =
        zipper::detail::extents::dynamic_extents_indices_v<extents_type>;

    return extents_type{std::array<index_type, dyn_indices.size()>{
        {internal_info::template extent<rank - 1 - dyn_indices[Ms]>(s)...}}};
  }

  static auto make_extents(const S &s) -> extents_type {
    constexpr auto dyn_indices =
        zipper::detail::extents::dynamic_extents_indices_v<extents_type>;
    return make_extents(
        s, std::make_integer_sequence<rank_type, dyn_indices.size()>{});
  }

  template <zipper::concepts::Index... Args>
  static auto get_value(auto &v, Args &&...args) -> auto & {
    return internal_info::get_value(v, std::forward<Args>(args)...);
  }
  template <zipper::concepts::Index... Args>
  static auto get_value(const auto &v, Args &&...args) -> const auto & {
    return internal_info::get_value(v, std::forward<Args>(args)...);
  }

  static auto initialize(const extents_type &e, const value_type &default_value)
      -> S {
    return detail::initialize(e, default_value);
  }
  static auto initialize(const value_type &default_value) -> S {
    return detail::initialize(default_value);
  }

  static void resize(S &s, const extents_type &e) {
    return detail::resize(s, e);
  }
};

template <typename S> StlStorageInfo(const S &) -> StlStorageInfo<S>;
} // namespace zipper::storage

#endif
