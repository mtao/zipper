#if !defined(ZIPPER_STORAGE_STLSTORAGEINFO_HPP)
#define ZIPPER_STORAGE_STLSTORAGEINFO_HPP
#include <array>
#include "zipper/detail/assert.hpp"
#include <vector>
#include <concepts>
#include <type_traits>

#include "zipper/concepts/Index.hpp"
#include "zipper/detail/extents/dynamic_extents_indices.hpp"
#include "zipper/types.hpp"

namespace zipper::storage {

// ---- Policy tags ------------------------------------------------------------
// Control whether zipper-like leaf types are unwrapped (their extents
// contribute to the overall StlStorageInfo extents) or treated as opaque scalar
// leaves (current default behavior).

/// Default policy: zipper types are opaque scalar leaves.
struct NoUnwrap {};

/// Unwrap policy: zipper types contribute their extents and value_type.
struct UnwrapZipper {};

namespace detail {

// ---- Structural concept for zipper-like types -------------------------------
// Uses duck typing so we don't need to include Vector.hpp / Matrix.hpp here.
template <typename T>
concept HasZipperInterface = requires(const std::remove_cvref_t<T> ct) {
  typename std::remove_cvref_t<T>::value_type;
  typename std::remove_cvref_t<T>::extents_type;
  requires(std::remove_cvref_t<T>::extents_type::rank() > 0);
  { ct.extent(rank_type{0}) } -> std::convertible_to<index_type>;
};

/// Whether a type T should be unwrapped given a policy.
template <typename T, typename Policy>
constexpr bool should_unwrap_v =
    std::same_as<Policy, UnwrapZipper> && HasZipperInterface<T>;

// ---- Leaf info helpers ------------------------------------------------------
// These provide the base-case behavior for the StlStorageInfo recursion.
// LeafInfo<T, false> = scalar leaf (rank 0, value_type = T).
// LeafInfo<T, true>  = zipper leaf (rank = T's rank, value_type = T's scalar).

template <typename T, bool IsZipperLeaf>
struct LeafInfo {
  using value_type = T;
  constexpr static rank_type rank = 0;
  static auto get_value(T &A) -> auto & { return A; }
  static auto get_value(const T &A) -> const auto & { return A; }

  static constexpr bool is_valid_extent(const T &) { return true; }

  static auto initialize(const value_type &t) -> value_type { return t; }
  static void resize(value_type &) {}
};

template <typename T>
struct LeafInfo<T, true> {
  using zipper_type = std::remove_cvref_t<T>;
  using value_type = typename zipper_type::value_type;
  using zipper_extents = typename zipper_type::extents_type;
  constexpr static rank_type rank = zipper_extents::rank();

  // get_value: forward remaining indices to the zipper's operator().
  // The args arrive in public (outermost-first) order, which is correct
  // for operator() on zipper types.
  template <typename... Args>
  static auto get_value(T &A, Args &&...args) -> decltype(auto) {
    return A(index_type(args)...);
  }
  template <typename... Args>
  static auto get_value(const T &A, Args &&...args) -> decltype(auto) {
    return A(index_type(args)...);
  }

  // Internal convention: extent 0 = innermost, extent (rank-1) = outermost.
  // Zipper convention: extent 0 = outermost (rows), extent (rank-1) = innermost (cols).
  // Map: internal N -> zipper (rank - 1 - N).
  template <rank_type N> consteval static auto static_extent() -> index_type {
    return zipper_extents::static_extent(rank - 1 - N);
  }

  template <rank_type M> static auto extent(const T &s) -> index_type {
    return s.extent(rank - 1 - M);
  }

  static constexpr bool is_valid_extent(const T &) { return true; }

  // initialize: create a zipper filled with the given value.
  template <typename... Args>
  static auto initialize(const value_type &t, Args &&... /*sizes*/) -> T {
    T result{};
    if constexpr (rank == 1) {
      for (index_type i = 0; i < result.extent(0); ++i) {
        result(i) = t;
      }
    } else if constexpr (rank == 2) {
      for (index_type i = 0; i < result.extent(0); ++i) {
        for (index_type j = 0; j < result.extent(1); ++j) {
          result(i, j) = t;
        }
      }
    } else {
      static_assert(rank <= 2,
                    "initialize for zipper leaves with rank > 2 not yet supported");
    }
    return result;
  }

  static void resize(T &) {}
};

// ---- StlStorageInfo ---------------------------------------------------------
// Internal helper for managing storage info.  Order of its extents are indexed
// in reverse to the final extents used because the left-most (0th extent) is
// the outermost structure.

// Primary template: leaf case (scalar or zipper leaf depending on Policy).
template <typename T, typename Policy = NoUnwrap>
struct StlStorageInfo : public LeafInfo<T, should_unwrap_v<T, Policy>> {};

// std::vector specialization
template <typename S, typename Policy>
struct StlStorageInfo<std::vector<S>, Policy>
    : public StlStorageInfo<S, Policy> {
  using child_info = StlStorageInfo<S, Policy>;
  using self_type = std::vector<S>;
  constexpr static rank_type rank = child_info::rank + 1;
  using value_type = typename child_info::value_type;
  constexpr static index_type my_static_extent = std::dynamic_extent;

  template <typename... Args>
  static auto get_value(std::vector<S> &A, index_type i, Args &&...args)
      -> decltype(auto) {
    return child_info::get_value(A[std::size_t{i}],
                                 std::forward<Args>(args)...);
  }
  template <typename... Args>
  static auto get_value(const std::vector<S> &A, index_type i, Args &&...args)
      -> decltype(auto) {
    return child_info::get_value(A[std::size_t{i}],
                                 std::forward<Args>(args)...);
  }

  template <rank_type N> consteval static auto static_extent() -> index_type {
    if constexpr (N + 1 == rank) {
      return my_static_extent;
    } else {
      return child_info::template static_extent<N>();
    }
  }

  template <rank_type M> static auto extent(const self_type &s) -> index_type {
    if constexpr (M + 1 == rank) {
      return s.size();
    } else if constexpr (s.size() == 0) {
      return 0;
    } else {
      return child_info::template extent<M>(s[0]);
    }
  }

  template <typename... Args>
  static constexpr auto is_valid_extent(const self_type &v, index_type size,
                                        Args &&...args) -> bool {
    if (v.size() != size) {
      return false;
    }
    for (const auto &x : v) {
      if (!child_info::is_valid_extent(x, std::forward<Args>(args)...)) {
        return false;
      }
    }
    return true;
  }

  template <typename... Args>
  static auto initialize(const value_type &d, index_type size, Args &&...args)
      -> self_type {
    self_type v(size, child_info::initialize(d, args...));
    return v;
  }

  template <typename... Args>
  static void resize(self_type &s, index_type size, Args &&...args) {
    s.resize(size);
    for (auto &x : s) {
      child_info::resize(x, args...);
    }
  }
};

// std::array specialization
template <size_t N, typename S, typename Policy>
struct StlStorageInfo<std::array<S, N>, Policy>
    : public StlStorageInfo<S, Policy> {
  using child_info = StlStorageInfo<S, Policy>;
  using self_type = std::array<S, N>;
  constexpr static rank_type rank = child_info::rank + 1;
  constexpr static index_type my_static_extent = index_type(N);
  using value_type = typename child_info::value_type;

  template <typename... Args>
  static auto get_value(std::array<S, N> &A, index_type i, Args &&...args)
      -> decltype(auto) {
    return child_info::get_value(A[std::size_t{i}],
                                 std::forward<Args>(args)...);
  }
  template <typename... Args>
  static auto get_value(const std::array<S, N> &A, index_type i,
                        Args &&...args) -> decltype(auto) {
    return child_info::get_value(A[std::size_t{i}],
                                 std::forward<Args>(args)...);
  }

  template <rank_type M> consteval static auto static_extent() -> index_type {
    if constexpr (M + 1 == rank) {
      return my_static_extent;
    } else {
      return child_info::template static_extent<M>();
    }
  }

  template <rank_type M>
  constexpr static auto extent(const self_type &s) -> index_type {
    if constexpr (M + 1 == rank) {
      return my_static_extent;
    } else if constexpr (N == 0) {
      return 0;
    } else {
      return child_info::template extent<M>(s[0]);
    }
  }

  template <typename... Args>
  static constexpr auto is_valid_extent(const self_type &v, index_type size,
                                        Args &&...args) -> bool {
    if (v.size() != size) {
      return false;
    }
    for (const auto &x : v) {
      if (!child_info::is_valid_extent(x, std::forward<Args>(args)...)) {
        return false;
      }
    }
    return true;
  }

  template <typename... Args>
  static auto initialize(const value_type &d, index_type size, Args &&...args)
      -> self_type {
    ZIPPER_ASSERT(size == my_static_extent);
    self_type v;
    for (auto &x : v) {
      x = child_info::initialize(d, args...);
    }
    return v;
  }
  static auto initialize(const value_type &d) -> self_type {
    self_type v;
    for (auto &x : v) {
      x = child_info::initialize(d);
    }
    return v;
  }

  template <typename... Args>
  static void resize(self_type &s, [[maybe_unused]] index_type size,
                     Args &&...args) {
    ZIPPER_ASSERT(size == my_static_extent);
    for (auto &x : s) {
      child_info::resize(x, args...);
    }
  }
};

} // namespace detail

// ---- Public StlStorageInfo --------------------------------------------------
// Wraps the internal detail version, reversing extent order so that the
// outermost structure is extent 0.

template <typename S, typename Policy = NoUnwrap>
struct StlStorageInfo : public detail::StlStorageInfo<S, Policy> {
  using internal_info = detail::StlStorageInfo<S, Policy>;

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
  static auto get_value(auto &v, Args &&...args) -> decltype(auto) {
    return internal_info::get_value(v, std::forward<Args>(args)...);
  }
  template <zipper::concepts::Index... Args>
  static auto get_value(const auto &v, Args &&...args) -> decltype(auto) {
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
