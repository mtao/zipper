#if !defined(ZIPPER_STORAGE_STLStorage_HPP)
#define ZIPPER_STORAGE_STLStorage_HPP
#include <array>
#include <cstdint>
#include <vector>

#include "zipper/concepts/IndexPackLike.hpp"
#include "zipper/detail/extents/dynamic_extents_indices.hpp"
#include "zipper/types.hpp"

namespace zipper::storage {

namespace detail {
// internal helper for managing storage info, order of its extents are indexed
// in reverse to the final extents used because the left-most (0th extent) is
// the outermost structure
template <typename T>
struct StlStorageInfo {
    using value_type = T;
    constexpr static rank_type rank = 0;
    static auto& get_value(T& A) { return A; }
    static const auto& get_value(const T& A) { return A; }
    template <rank_type N>
    using get_type = std::enable_if<(N == -1), value_type>;

    consteval bool is_valid_extent(const T&) { return true; }
};
template <typename S>
struct StlStorageInfo<std::vector<S>> : public StlStorageInfo<S> {
    using self_type = std::vector<S>;
    constexpr static rank_type rank = StlStorageInfo<S>::rank + 1;
    constexpr static index_type my_static_extent = std::dynamic_extent;
    template <typename... Args>
    static auto& get_value(std::vector<S>& A, index_type i, Args&&... args) {
        return StlStorageInfo<S>::get_value(A[i], std::forward<Args>(args)...);
    }
    template <typename... Args>
    static const auto& get_value(const std::vector<S>& A, index_type i,
                                 Args&&... args) {
        return StlStorageInfo<S>::get_value(A[i], std::forward<Args>(args)...);
    }

    template <rank_type N>
        requires(rank >= 0)
    using get_type =
        std::conditional_t<N + 1 == rank, self_type,
                           typename StlStorageInfo<S>::template get_type<N>>;

    template <rank_type N>
    consteval static index_type static_extent() {
        if constexpr (N + 1 == rank) {
            return my_static_extent;
        } else {
            return StlStorageInfo<get_type<N>>::template static_extent<N>();
        }
    }
    template <rank_type M>
    static index_type extent(const self_type& s) {
        if constexpr (M + 1 == rank) {
            return s.size();
        } else if constexpr (s.size() == 0) {
            return 0;
        } else {
            return StlStorageInfo<S>::template extent<M>(s[0]);
        }
    }
    template <typename... Args>
    consteval bool is_valid_extent(const self_type& v, index_type size,
                                   Args&&... args) {
        if (v.size() != size) {
            return false;
        }
        for (const auto& x : v) {
            if (!StlStorageInfo<S>::is_valid_extent(
                    x, std::forward<Args>(args)...)) {
                return false;
            }
        }
        return true;
    }
};
template <size_t N, typename S>
struct StlStorageInfo<std::array<S, N>> : public StlStorageInfo<S> {
    using self_type = std::array<S, N>;
    constexpr static rank_type rank = StlStorageInfo<S>::rank + 1;
    constexpr static index_type my_static_extent = index_type(N);
    template <typename... Args>
    static auto& get_value(std::array<S, N>& A, index_type i, Args&&... args) {
        return StlStorageInfo<S>::get_value(A[i], std::forward<Args>(args)...);
    }
    template <typename... Args>
    static const auto& get_value(const std::array<S, N>& A, index_type i,
                                 Args&&... args) {
        return StlStorageInfo<S>::get_value(A[i], std::forward<Args>(args)...);
    }
    template <rank_type M>
        requires(rank >= 0)
    using get_type =
        std::conditional_t<M + 1 == rank, self_type,
                           typename StlStorageInfo<S>::template get_type<M>>;

    template <rank_type M>
    consteval static index_type static_extent() {
        if constexpr (M + 1 == rank) {
            return my_static_extent;
        } else {
            return StlStorageInfo<get_type<M>>::template static_extent<M>();
        }
    }
    template <rank_type M>
    constexpr static index_type extent(const self_type& s) {
        if constexpr (M + 1 == rank) {
            return my_static_extent;
            ;
        } else if constexpr (N == 0) {
            return 0;
        } else {
            return StlStorageInfo<S>::template extent<M>(s[0]);
        }
    }
    template <typename>
    struct detail_;
    template <rank_type... Ms>
    struct detail_<std::integer_sequence<rank_type, Ms...>> {
        //
    };
    using detail =
        detail_<decltype(std::make_integer_sequence<rank_type, rank>{})>;

    template <typename... Args>
    consteval bool is_valid_extent(const self_type& v, index_type size,
                                   Args&&... args) {
        if (v.size() != size) {
            return false;
        }
        for (const auto& x : v) {
            if (!StlStorageInfo<S>::is_valid_extent(
                    x, std::forward<Args>(args)...)) {
                return false;
            }
        }
        return true;
    }
    // using extents_type = zipper::extents<N>();
    // constexpr static extents_type extents(const std::array<int64_t, N>& A) {
    //     return extents_type();
    // }
};
// using extents_type =

}  // namespace detail

template <typename S>
struct StlStorageInfo : public detail::StlStorageInfo<S> {
    using internal_info = detail::StlStorageInfo<S>;

    constexpr static rank_type rank = internal_info::rank;
    template <typename>
    struct detail_;
    template <rank_type... Ms>
    struct detail_<std::integer_sequence<rank_type, Ms...>> {
        using extents_type =
            extents<internal_info::template static_extent<rank - 1 - Ms>()...>;
        //
    };
    using detail =
        detail_<decltype(std::make_integer_sequence<rank_type, rank>{})>;
    using extents_type = typename detail::extents_type;

    template <rank_type... Ms>
    static extents_type make_extents(const S& s,
                                     std::integer_sequence<rank_type, Ms...>) {
        constexpr auto dyn_indices =
            zipper::detail::extents::dynamic_extents_indices_v<extents_type>;

        return extents_type{std::array<index_type, dyn_indices.size()>{
            {internal_info::template extent<rank - 1 - dyn_indices[Ms]>(
                s)...}}};
    }

    static extents_type make_extents(const S& s) {
        constexpr auto dyn_indices =
            zipper::detail::extents::dynamic_extents_indices_v<extents_type>;
        return make_extents(
            s, std::make_integer_sequence<rank_type, dyn_indices.size()>{});
    }

    template <zipper::concepts::IndexLike... Args>
    static auto& get_value(auto& v, Args&&... args) {
        return internal_info::get_value(v, std::forward<Args>(args)...);
    }
    template <zipper::concepts::IndexLike... Args>
    static const auto& get_value(const auto& v, Args&&... args) {
        return internal_info::get_value(v, std::forward<Args>(args)...);
    }
};

template <typename S>
struct StlStorage {
   public:
    using info_helper = StlStorageInfo<S>;
    using extents_type = typename info_helper::extents_type;
    using value_type = typename info_helper::value_type;
    consteval static index_type rank() { return info_helper::rank(); }
    consteval static index_type static_extent(rank_type r) {
        return extents_type::static_extent;
    }
    StlStorage(const S& d) : m_data(d) {}

    extents_type extents() const { return info_helper::make_extents(m_data); }

    template <zipper::concepts::IndexLike... Args>
    const value_type& const_coeff_ref(Args&&... args) const {
        return info_helper::get_value(m_data, std::forward<Args>(args)...);
    }
    template <zipper::concepts::IndexLike... Args>
    value_type& coeff_ref(Args&&... args) {
        return info_helper::get_value(m_data, std::forward<Args>(args)...);
    }
    template <zipper::concepts::IndexLike... Args>
    value_type coeff(Args&&... args) const {
        return info_helper::get_value(m_data, std::forward<Args>(args)...);
    }

   private:
    S m_data;
};

template <typename S>
StlStorage(const S&) -> StlStorage<S>;
template <typename S>
StlStorageInfo(const S&) -> StlStorageInfo<S>;

}  // namespace zipper::storage

#endif
