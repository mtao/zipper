#if !defined(ZIPPER_VIEWS_VIEWBASE_HPP)
#define ZIPPER_VIEWS_VIEWBASE_HPP
#include "zipper/types.hpp"
#include <type_traits>

#include "detail/ViewTraits.hpp"
#include "zipper/concepts/IndexPackLike.hpp"
#include "zipper/concepts/SlicePackLike.hpp"
#include "zipper/concepts/TupleLike.hpp"
#include "zipper/concepts/ViewAccessTuple.hpp"
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/detail//ExtentsTraits.hpp"

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
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    // view does not permute underlying value of indices, so coefficient-wise
    // operations are valid
    constexpr static bool is_coefficient_consistent =
        traits::is_coefficient_consistent;
    constexpr static bool is_writable = traits::is_writable;
    constexpr static rank_type rank = extents_type::rank();
    static_assert(extents_type::rank() >= 0);
    using array_type = std::array<index_type, rank>;
    // using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    constexpr index_type extent(rank_type i) const {
        return extents().extent(i);
    }
    static consteval index_type static_extent(rank_type i) {
        return extents_type::static_extent(i);
    }
    constexpr auto extents() const -> const extents_type&{ return derived().extents(); }

   public:
    template <typename... Indices>
    auto coeff(Indices&&... indices) const -> value_type
        requires((concepts::IndexLike<std::decay_t<Indices>> && ...));
    template <typename... Indices>
    auto coeff_ref(Indices&&... indices) -> value_type& requires(
        is_writable && (concepts::IndexLike<std::decay_t<Indices>> && ...));

    template <typename... Indices>
    auto const_coeff_ref(Indices&&... indices) const
        -> const value_type& requires(
            is_writable && (concepts::IndexLike<std::decay_t<Indices>> && ...));

    template <typename... Args>
    auto operator()(Args&&... idxs) const -> decltype(auto);

    template <typename... Args>
    auto operator()(Args&&... idxs) -> decltype(auto)
        requires(is_writable);

    template <typename... Args>
    auto access_pack(Args&&... idxs) const -> decltype(auto);

    template <typename... Args>
    auto access_index_pack(Args&&... idxs) const -> decltype(auto)
        requires(concepts::SlicePackLike<Args...>);

    template <typename... Args>
    auto access_pack(Args&&... idxs) -> decltype(auto);

    template <typename... Args>
    auto access_index_pack(Args&&... idxs) -> decltype(auto)
        requires(concepts::SlicePackLike<Args...>);

    template <typename... Slices>
    auto access_slice(Slices&&... slices) const
        requires(concepts::SlicePackLike<Slices...> &&
                 !concepts::IndexPackLike<Slices...>);

    template <concepts::ViewAccessTuple Tuple, std::size_t... N>
    auto access_tuple(const Tuple& t, std::index_sequence<N...>) const
        -> decltype(auto);
    template <concepts::ViewAccessTuple Tuple>
    auto access_tuple(const Tuple& t) const -> decltype(auto);

    template <typename... Slices>
    auto access_slice(Slices&&... slices)
        requires(concepts::SlicePackLike<Slices...> &&
                 !concepts::IndexPackLike<Slices...>);

    template <concepts::ViewAccessTuple Tuple, std::size_t... N>
    auto access_tuple(const Tuple& t, std::index_sequence<N...>)
        -> decltype(auto);
    template <concepts::ViewAccessTuple Tuple>
    auto access_tuple(const Tuple& t) -> decltype(auto);

    constexpr size_t size() const { return extents_traits::size(extents()); }
};

}  // namespace zipper::views
#include "ViewBase.hxx"
#endif

