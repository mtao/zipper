#if !defined(UVL_VIEWS_VIEWBASE_HPP)
#define UVL_VIEWS_VIEWBASE_HPP
#include <experimental/mdspan>

#include "uvl/concepts/IndexPackLike.hpp"
#include "uvl/concepts/SlicePackLike.hpp"
#include "detail/ViewTraits.hpp"
#include "uvl/concepts/TupleLike.hpp"
#include "uvl/detail/ExtentsTraits.hpp"
#include "uvl/detail/tuple_size.hpp"

namespace uvl::views {

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
    constexpr static bool is_coefficient_consistent = traits::is_coefficient_consistent;
    constexpr static bool is_writable = traits::is_writable;
    constexpr static rank_type rank = extents_type::rank();
    using array_type = std::array<index_type, rank>;
    // using extents_traits = uvl::detail::ExtentsTraits<extents_type>;

    constexpr index_type extent(rank_type i) const {
        return extents().extent(i);
    }
    const extents_type& extents() const { return derived().extents(); }

   public:
    template <std::size_t... Idxs>
    auto _coeff(concepts::TupleLike auto const& t,
                std::integer_sequence<index_type, Idxs...>) const
        -> value_type {
        return derived().coeff(std::get<Idxs>(t)...);
    }

    template <typename... Indices>
    auto coeff(Indices&&... indices) const -> value_type {
        if constexpr (sizeof...(Indices) == 1 &&
                      (concepts::TupleLike<std::decay_t<Indices>> && ...)) {
            static_assert(
                ((uvl::detail::tuple_size<std::decay_t<Indices>>::value ==
                  rank) &&
                 ...));
            return _coeff(indices...,
                          std::make_integer_sequence<std::size_t, rank>{});
        } else if constexpr ((std::is_integral_v<std::decay_t<Indices>> &&
                              ...)) {
            static_assert((!concepts::TupleLike<std::decay_t<Indices>> && ...));
            static_assert(sizeof...(Indices) == rank);
            return derived().coeff(std::forward<Indices>(indices)...);
        }
    }

    template <std::size_t... Idxs>
    auto _const_coeff_ref(concepts::TupleLike auto const& t,
                          std::integer_sequence<index_type, Idxs...>) const
        -> const value_type& requires(is_writable) {
            return derived().const_coeff_ref(std::get<Idxs>(t)...);
        }

    template <typename... Indices>
    auto const_coeff_ref(
        Indices&&... indices) const -> const value_type& requires(is_writable) {
        if constexpr (sizeof...(Indices) == 1 &&
                      (concepts::TupleLike<std::decay_t<Indices>> && ...)) {
            static_assert(
                ((uvl::detail::tuple_size<std::decay_t<Indices>>::value ==
                  rank) &&
                 ...));
            return _const_coeff_ref(
                indices..., std::make_integer_sequence<std::size_t, rank>{});
        } else if constexpr ((std::is_integral_v<std::decay_t<Indices>> &&
                              ...)) {
            static_assert((!concepts::TupleLike<std::decay_t<Indices>> && ...));
            static_assert(sizeof...(Indices) == rank);
            return derived().const_coeff_ref(std::forward<Indices>(indices)...);
        }
    }

    template <std::size_t... Idxs>
    auto _coeff_ref(concepts::TupleLike auto const& t,
                    std::integer_sequence<index_type, Idxs...>)
        -> value_type& requires(is_writable) {
            return derived().coeff_ref(std::get<Idxs>(t)...);
        }

    template <typename... Indices>
    auto coeff_ref(Indices&&... indices) -> value_type& requires(is_writable) {
        if constexpr (sizeof...(Indices) == 1 &&
                      (concepts::TupleLike<std::decay_t<Indices>> && ...)) {
            static_assert(
                ((uvl::detail::tuple_size<std::decay_t<Indices>>::value ==
                  rank) &&
                 ...));
            return _coeff_ref(indices...,
                              std::make_integer_sequence<std::size_t, rank>{});
        } else if constexpr ((std::is_integral_v<std::decay_t<Indices>> &&
                              ...)) {
            static_assert((!concepts::TupleLike<std::decay_t<Indices>> && ...));
            static_assert(sizeof...(Indices) == rank);
            return derived().coeff_ref(std::forward<Indices>(indices)...);
        }
    }

    template <typename... Args>
    auto access(Args&&... idxs) const -> decltype(auto)

    {
        if constexpr (is_writable) {
            return const_coeff_ref(std::forward<Args>(idxs)...);
        } else {
            return coeff(std::forward<Args>(idxs)...);
        }
    }
    template <typename... Args>
    auto access(Args&&... idxs) -> decltype(auto)
        requires(is_writable)

    {
        return coeff_ref(std::forward<Args>(idxs)...);
    }

    template <typename... Args>
    auto operator()(Args&&... idxs) const -> decltype(auto)

    {
        //if constexpr(concepts::IndexPackLike<std::decay_t<Args>...>) {
            return access(std::forward<Args>(idxs)...);
        //} else {
        //}

    }
    template <typename... Args>
    auto operator()(Args&&... idxs) -> decltype(auto)
        requires(is_writable)

    {
        return access(std::forward<Args>(idxs)...);
    }

    template <typename... Args>
    auto access_slice(Args&&... idxs) const -> decltype(auto)
    rqeuires (concepts::SlicePackLikej

    {
        if constexpr (is_writable) {
            return const_coeff_ref(std::forward<Args>(idxs)...);
        } else {
            return coeff(std::forward<Args>(idxs)...);
        }
    }
};
}  // namespace uvl::views
#endif
