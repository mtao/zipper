#pragma once

#include "zipper/concepts/Zipper.hpp"
#include "zipper/expression/concepts/capabilities.hpp"
#include "zipper/detail/NoAliasTraits.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"
#include <type_traits>
#include <utility>

namespace zipper::detail {

// Fresh owning storage cannot alias a source, so construction may evaluate
// straight into it (see ZipperBase's converting constructor).
template <typename T>
inline constexpr bool is_fresh_mdarray_v = false;

template <typename T, typename E, typename L, typename A>
inline constexpr bool is_fresh_mdarray_v<expression::nullary::MDArray<T, E, L, A>> = true;

// Which destination expressions support noalias is opted into by each
// expression's own header via supports_noalias_v (see NoAliasTraits.hpp).

template <typename Destination>
concept NoAliasDestination =
    concepts::Zipper<Destination> &&
    !std::is_const_v<std::remove_reference_t<Destination>> &&
    requires(Destination &dest) {
        requires expression::concepts::WritableExpression<decltype(dest.expression())>;
        requires(!std::is_const_v<std::remove_reference_t<decltype(dest.expression())>>);
        requires supports_noalias_v<std::remove_cvref_t<decltype(dest.expression())>>;
    };

/// Assignment-only proxy: the caller promises that no RHS read (including
/// callback captures) aliases the destination, even across a destination resize.
template <NoAliasDestination Destination>
class NoAliasProxy {
public:
    explicit NoAliasProxy(Destination &destination) : m_destination(destination) {}

    template <concepts::Expression From>
        requires(utils::extents::assignable_extents_v<
                 typename From::extents_type, typename Destination::extents_type>)
    auto operator=(const From &from) -> Destination & {
        using to_type = typename Destination::expression_type;
        expression::detail::AssignHelper<From, to_type>::assign_independent(
            from, m_destination.expression());
        return m_destination;
    }

    template <concepts::Zipper From>
        requires(utils::extents::assignable_extents_v<
                 typename From::extents_type, typename Destination::extents_type>)
    auto operator=(const From &from) -> Destination & {
        return *this = from.expression();
    }

private:
    Destination &m_destination;
};

} // namespace zipper::detail
