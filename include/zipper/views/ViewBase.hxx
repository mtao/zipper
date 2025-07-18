#if !defined(ZIPPER_VIEWS_VIEWBASE_HXX)
#define ZIPPER_VIEWS_VIEWBASE_HXX

#include "ViewBase.hpp"
#include "zipper/utils/extents/indices_in_range.hpp"
#include "zipper/detail/tuple_size.hpp"

namespace zipper::views {

template <typename Derived>
template <typename... Indices>
auto ViewBase<Derived>::coeff(Indices&&... indices) const -> value_type
    requires((concepts::IndexLike<std::decay_t<Indices>> && ...))
{
    constexpr static rank_type size = sizeof...(Indices);
    // rank 0 views act like scalar values
    static_assert(rank == 0 || size == rank);
    return derived().coeff(std::forward<Indices>(indices)...);
}
template <typename Derived>
template <typename... Indices>
auto ViewBase<Derived>::coeff_ref(Indices&&... indices) -> value_type&
    requires(is_writable && (concepts::IndexLike<std::decay_t<Indices>> && ...))
{
    constexpr static rank_type size = sizeof...(Indices);
    // rank 0 views act like scalar values
    static_assert(rank == 0 || size == rank);
    return derived().coeff_ref(std::forward<Indices>(indices)...);
}

template <typename Derived>
template <typename... Indices>
auto ViewBase<Derived>::const_coeff_ref(Indices&&... indices) const
    -> const value_type&
    requires(is_writable && (concepts::IndexLike<std::decay_t<Indices>> && ...))
{
    constexpr static rank_type size = sizeof...(Indices);
    // rank 0 views act like scalar values
    static_assert(rank == 0 || size == rank);
    return derived().const_coeff_ref(std::forward<Indices>(indices)...);
}

template <typename Derived>
template <typename... Args>
auto ViewBase<Derived>::operator()(Args&&... idxs) const -> decltype(auto)

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
                decltype(auto) v = access_tuple(std::forward<Args>(idxs)...);
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
template <typename Derived>
template <typename... Args>
auto ViewBase<Derived>::operator()(Args&&... idxs) -> decltype(auto)
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
                decltype(auto) v = access_tuple(std::forward<Args>(idxs)...);
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

template <typename Derived>
template <typename... Args>
auto ViewBase<Derived>::access_pack(Args&&... idxs) const -> decltype(auto)

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

template <typename Derived>
template <typename... Args>
auto ViewBase<Derived>::access_index_pack(Args&&... idxs) const
    -> decltype(auto)
    requires(concepts::SlicePackLike<Args...>)
{
#if !defined(NDEBUG)
    zipper::utils::extents::indices_in_range(extents(), idxs...);
#endif
    if constexpr (is_writable) {
        return const_coeff_ref(std::forward<Args>(idxs)...);
    } else {
        return coeff(std::forward<Args>(idxs)...);
    }
}

template <typename Derived>
template <typename... Args>
auto ViewBase<Derived>::access_pack(Args&&... idxs) -> decltype(auto)

{
    static_assert(concepts::IndexPackLike<Args...> ||
                  concepts::SlicePackLike<Args...>);
    if constexpr (concepts::IndexPackLike<Args...>) {
        return access_index_pack(std::forward<Args>(idxs)...);
    } else if constexpr (concepts::SlicePackLike<Args...>) {
        return access_slice(std::forward<Args>(idxs)...);
    }
}

template <typename Derived>
template <typename... Args>
auto ViewBase<Derived>::access_index_pack(Args&&... idxs) -> decltype(auto)
    requires(concepts::SlicePackLike<Args...>)
{
#if !defined(NDEBUG)
    zipper::utils::extents::indices_in_range(extents(), idxs...);
#endif
    return coeff_ref(std::forward<Args>(idxs)...);
}

template <typename Derived>
template <concepts::ViewAccessTuple Tuple, std::size_t... N>
auto ViewBase<Derived>::access_tuple(const Tuple& t,
                                     std::index_sequence<N...>) const
    -> decltype(auto) {
    return access_pack(std::get<N>(t)...);
}
template <typename Derived>
template <concepts::ViewAccessTuple Tuple>
auto ViewBase<Derived>::access_tuple(const Tuple& t) const -> decltype(auto) {
    return access_tuple(
        t, std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>{});
}

template <typename Derived>
template <concepts::ViewAccessTuple Tuple, std::size_t... N>
auto ViewBase<Derived>::access_tuple(const Tuple& t, std::index_sequence<N...>)
    -> decltype(auto) {
    return access_pack(std::get<N>(t)...);
}
template <typename Derived>
template <concepts::ViewAccessTuple Tuple>
auto ViewBase<Derived>::access_tuple(const Tuple& t) -> decltype(auto) {
    return access_tuple(
        t, std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>{});
}

namespace unary {
template <concepts::QualifiedViewDerived ViewType, typename... Slices>
    requires(concepts::SlicePackLike<Slices...>)
class SliceView;
}
template <typename Derived>
template <typename... Slices>
auto ViewBase<Derived>::access_slice(Slices&&... slices) const
    requires(concepts::SlicePackLike<Slices...> &&
             !concepts::IndexPackLike<Slices...>)
{
    return unary::SliceView<const Derived, std::decay_t<Slices>...>(
        derived(), std::forward<Slices>(slices)...);
}

template <typename Derived>
template <typename... Slices>
auto ViewBase<Derived>::access_slice(Slices&&... slices)
    requires(concepts::SlicePackLike<Slices...> &&
             !concepts::IndexPackLike<Slices...>)
{
    return unary::SliceView<Derived, std::decay_t<Slices>...>(
        derived(), std::forward<Slices>(slices)...);

    // return unary::SliceView<Derived, false, std::decay_t<Slices>...>(
    //     derived(), slices...);
}
}  // namespace zipper::views
#endif

