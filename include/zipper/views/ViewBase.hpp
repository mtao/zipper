#if !defined(ZIPPER_VIEWS_VIEWBASE_HPP)
#define ZIPPER_VIEWS_VIEWBASE_HPP
#include <experimental/mdspan>
#include <type_traits>

#include "zipper/concepts/ViewDerived.hpp"
// #include "zipper/views/unary/SliceView.hpp"
#include "detail/ViewTraits.hpp"
#include "zipper/concepts/IndexPackLike.hpp"
#include "zipper/concepts/SlicePackLike.hpp"
#include "zipper/concepts/TupleLike.hpp"
#include "zipper/concepts/ViewAccessTuple.hpp"
#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/detail/extents/indices_in_range.hpp"
#include "zipper/detail/tuple_size.hpp"

namespace zipper::views {

template <typename Derived_>
class ViewBase {
   public:
    using Derived = Derived_;

    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }

    using traits = detail::ViewTraits<Derived>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    constexpr static bool is_coefficient_consistent =
        traits::is_coefficient_consistent;
    constexpr static bool is_writable = traits::is_writable;
    constexpr static rank_type rank = extents_type::rank();
    using array_type = std::array<index_type, rank>;
    // using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    constexpr index_type extent(rank_type i) const {
        return extents().extent(i);
    }
    const extents_type& extents() const { return derived().extents(); }

   public:
    template <typename... Indices>
    auto coeff(Indices&&... indices) const -> value_type
        requires((concepts::IndexLike<std::decay_t<Indices>> && ...))
    {
        constexpr static rank_type size = sizeof...(Indices);
        static_assert(rank == 0 || size == rank);
        return derived().coeff(std::forward<Indices>(indices)...);
    }
    template <typename... Indices>
    auto coeff_ref(Indices&&... indices) -> value_type& requires(
        is_writable && (concepts::IndexLike<std::decay_t<Indices>> && ...)) {
        constexpr static rank_type size = sizeof...(Indices);
        static_assert(rank == 0 || size == rank);
        return derived().coeff_ref(std::forward<Indices>(indices)...);
    }

    template <typename... Indices>
    auto const_coeff_ref(Indices&&... indices) const
        -> const value_type& requires(
            is_writable &&
            (concepts::IndexLike<std::decay_t<Indices>> && ...)) {
            constexpr static rank_type size = sizeof...(Indices);
            static_assert(rank == 0 || size == rank);
            return derived().const_coeff_ref(std::forward<Indices>(indices)...);
        }

    template <typename... Args>
    auto operator()(Args&&... idxs) const -> decltype(auto)

    {
        if constexpr (extents_type::rank() == 0) {
            if constexpr (sizeof...(Args) == 0) {
                decltype(auto) v = access_index_pack();
                static_assert(!std::is_void_v<std::decay_t<decltype(v)>>);
                return v;
            } else {
                static_assert(sizeof...(Args) == 1);
                static_assert((concepts::TupleLike<std::decay_t<Args>> && ...));

                if constexpr (((std::tuple_size_v<std::decay_t<Args>> == 0) &&
                               ... && true)) {
                    decltype(auto) v = access_index_pack();
                    static_assert(!std::is_void_v<std::decay_t<decltype(v)>>);
                    return v;
                } else {
                    decltype(auto) v =
                        access_tuple(std::forward<Args>(idxs)...);
                    static_assert(!std::is_void_v<std::decay_t<decltype(v)>>);
                    return v;
                }
            }

        } else if constexpr (sizeof...(Args) == 1 &&
                             (concepts::ViewAccessTuple<std::decay_t<Args>> &&
                              ...)) {
            decltype(auto) v = access_tuple(std::forward<Args>(idxs)...);
            static_assert(!std::is_void_v<std::decay_t<decltype(v)>>);
            return v;
        } else {
            decltype(auto) v = access_pack(std::forward<Args>(idxs)...);
            static_assert(!std::is_void_v<std::decay_t<decltype(v)>>);
            return v;
        }
    }
    template <typename... Args>
    auto operator()(Args&&... idxs) -> decltype(auto)
        requires(is_writable)

    {
        if constexpr (extents_type::rank() == 0) {
            if constexpr (sizeof...(Args) == 0) {
                decltype(auto) v = access_index_pack();
                static_assert(!std::is_void_v<std::decay_t<decltype(v)>>);
                return v;
            } else {
                static_assert(sizeof...(Args) == 1);
                static_assert((concepts::TupleLike<std::decay_t<Args>> && ...));

                if constexpr (((std::tuple_size_v<std::decay_t<Args>> == 0) &&
                               ... && true)) {
                    decltype(auto) v = access_index_pack();
                    static_assert(!std::is_void_v<std::decay_t<decltype(v)>>);
                    return v;
                } else {
                    decltype(auto) v =
                        access_tuple(std::forward<Args>(idxs)...);
                    static_assert(!std::is_void_v<std::decay_t<decltype(v)>>);
                    return v;
                }
            }

        } else if constexpr (sizeof...(Args) == 1 &&
                             (concepts::ViewAccessTuple<std::decay_t<Args>> &&
                              ...)) {
            decltype(auto) v = access_tuple(std::forward<Args>(idxs)...);
            static_assert(!std::is_void_v<std::decay_t<decltype(v)>>);
            return v;
        } else {
            decltype(auto) v = access_pack(std::forward<Args>(idxs)...);
            static_assert(!std::is_void_v<std::decay_t<decltype(v)>>);
            return v;
        }
    }

    template <typename... Args>
    auto access_pack(Args&&... idxs) const -> decltype(auto)

    {
        static_assert(!(concepts::ViewAccessTuple<std::decay_t<Args>> && ...));
        static_assert(!(concepts::TupleLike<std::decay_t<Args>> && ...));
        // static_assert(concepts::IndexPackLike<Args...> ||
        //               concepts::SlicePackLike<Args...>);
        if constexpr (concepts::IndexPackLike<Args...>) {
            return access_index_pack(std::forward<Args>(idxs)...);
        } else if constexpr (concepts::SlicePackLike<Args...>) {
            return access_slice(std::forward<Args>(idxs)...);
        }
    }

    template <typename... Args>
    auto access_index_pack(Args&&... idxs) const -> decltype(auto)
        requires(concepts::SlicePackLike<Args...>)
    {
#if !defined(NDEBUG)
        zipper::detail::extents::indices_in_range(extents(), idxs...);
#endif
        if constexpr (is_writable) {
            return const_coeff_ref(std::forward<Args>(idxs)...);
        } else {
            return coeff(std::forward<Args>(idxs)...);
        }
    }

    template <typename... Args>
    auto access_pack(Args&&... idxs) -> decltype(auto)

    {
        static_assert(concepts::IndexPackLike<Args...> ||
                      concepts::SlicePackLike<Args...>);
        if constexpr (concepts::IndexPackLike<Args...>) {
            return access_index_pack(std::forward<Args>(idxs)...);
        } else if constexpr (concepts::SlicePackLike<Args...>) {
            return access_slice(std::forward<Args>(idxs)...);
        }
    }

    template <typename... Args>
    auto access_index_pack(Args&&... idxs) -> decltype(auto)
        requires(concepts::SlicePackLike<Args...>)
    {
#if !defined(NDEBUG)
        zipper::detail::extents::indices_in_range(extents(), idxs...);
#endif
        return coeff_ref(std::forward<Args>(idxs)...);
    }

    template <typename... Slices>
    auto access_slice(Slices&&... slices) const
        requires(concepts::SlicePackLike<Slices...> &&
                 !concepts::IndexPackLike<Slices...>);

    template <concepts::ViewAccessTuple Tuple, std::size_t... N>
    auto access_tuple(const Tuple& t, std::index_sequence<N...>) const
        -> decltype(auto) {
        return access_pack(std::get<N>(t)...);
    }
    template <concepts::ViewAccessTuple Tuple>
    auto access_tuple(const Tuple& t) const -> decltype(auto) {
        return access_tuple(
            t,
            std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>{});
    }

    template <typename... Slices>
    auto access_slice(Slices&&... slices)
        requires(concepts::SlicePackLike<Slices...> &&
                 !concepts::IndexPackLike<Slices...>);

    template <concepts::ViewAccessTuple Tuple, std::size_t... N>
    auto access_tuple(const Tuple& t, std::index_sequence<N...>)
        -> decltype(auto) {
        return access_pack(std::get<N>(t)...);
    }
    template <concepts::ViewAccessTuple Tuple>
    auto access_tuple(const Tuple& t) -> decltype(auto) {
        return access_tuple(
            t,
            std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>{});
    }
};

namespace unary {
template <concepts::ViewDerived ViewType, bool IsConst, typename... Slices>
    requires(concepts::SlicePackLike<Slices...>)
class SliceView;
}
template <typename Derived>
template <typename... Slices>
auto ViewBase<Derived>::access_slice(Slices&&... slices) const
    requires(concepts::SlicePackLike<Slices...> &&
             !concepts::IndexPackLike<Slices...>)
{
    return unary::SliceView<Derived, true, std::decay_t<Slices>...>(
        derived(), std::forward<Slices>(slices)...);
}

template <typename Derived>
template <typename... Slices>
auto ViewBase<Derived>::access_slice(Slices&&... slices)
    requires(concepts::SlicePackLike<Slices...> &&
             !concepts::IndexPackLike<Slices...>)
{
    return unary::SliceView<Derived, false, std::decay_t<Slices>...>(
        derived(), std::forward<Slices>(slices)...);

    // return unary::SliceView<Derived, false, std::decay_t<Slices>...>(
    //     derived(), slices...);
}
}  // namespace zipper::views
#endif

