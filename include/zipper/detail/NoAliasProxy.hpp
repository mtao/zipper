#pragma once

#include "zipper/concepts/Zipper.hpp"
#include "zipper/expression/concepts/capabilities.hpp"
#include "zipper/detail/NoAliasTraits.hpp"
#include "zipper/static_scalar.hpp"
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

    /// `dest.noalias() += src` / `-= src`: dest + src without the aliasing
    /// check (accumulable sources, e.g. a matrix product, add themselves in).
    template <concepts::Zipper From>
    auto operator+=(const From &from) -> Destination &
        requires requires(const Destination &d, const From &f) { d + f; }
    {
        accumulate(from, cw<1>);
        return m_destination;
    }
    template <concepts::Zipper From>
    auto operator-=(const From &from) -> Destination &
        requires requires(const Destination &d, const From &f) { d - f; }
    {
        accumulate(from, cw<-1>);
        return m_destination;
    }

private:
    template <typename From,
              concepts::Coefficient<typename Destination::value_type> Alpha>
    void accumulate(const From &from, Alpha alpha) {
        using to_type = typename Destination::expression_type;
        expression::detail::AssignHelper<typename From::expression_type, to_type>::
            accumulate_independent(from.expression(), m_destination.expression(), alpha);
    }

    Destination &m_destination;
};

} // namespace zipper::detail
