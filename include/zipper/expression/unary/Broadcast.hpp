#if !defined(ZIPPER_EXPRESSION_UNARY_BROADCAST_HPP)
#define ZIPPER_EXPRESSION_UNARY_BROADCAST_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Extents.hpp"
#include "detail/invert_integer_sequence.hpp"
#include "zipper/detail/pack_index.hpp"

#include <utility>

namespace zipper::expression {
namespace unary {
/// The child repeated along new axes `Axes...` (positions in the result),
/// like NumPy broadcasting: result(i...) = child(i with the new axes
/// dropped).  The result's extents (static or dynamic) are given; on the
/// child's axes they match the child.
template <zipper::concepts::QualifiedExpression Child, zipper::concepts::Extents Extents, rank_type... Axes>
class Broadcast;
} // namespace unary

template <zipper::concepts::QualifiedExpression Child, zipper::concepts::Extents Extents, rank_type... Axes>
struct detail::ExpressionDetail<unary::Broadcast<Child, Extents, Axes...>> {
    using child_traits = detail::ExpressionTraits<std::decay_t<Child>>;
    constexpr static rank_type rank = child_traits::extents_type::rank() + sizeof...(Axes);
    static_assert(Extents::rank() == rank,
                  "broadcast: the result has the child's axes plus one per new axis");
    // Maps the child's axes to their positions in the result.
    using axis_map = unary::detail::invert_integer_sequence<rank, Axes...>;
};

template <zipper::concepts::QualifiedExpression Child, zipper::concepts::Extents Extents, rank_type... Axes>
struct detail::ExpressionTraits<unary::Broadcast<Child, Extents, Axes...>>
  : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
        Child,
        zipper::detail::AccessFeatures::const_value()> {
    using _Detail = detail::ExpressionDetail<unary::Broadcast<Child, Extents, Axes...>>;
    using extents_type = Extents;
    // Many coefficients read one child coefficient.
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace unary {
template <zipper::concepts::QualifiedExpression Child, zipper::concepts::Extents Extents, rank_type... Axes>
class Broadcast : public UnaryExpressionBase<Broadcast<Child, Extents, Axes...>, Child> {
  public:
    using self_type = Broadcast<Child, Extents, Axes...>;
    using Base = UnaryExpressionBase<self_type, Child>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using detail_type = zipper::expression::detail::ExpressionDetail<self_type>;
    using extents_type = typename traits::extents_type;
    using value_type = typename traits::value_type;
    using Base::expression;

    template <typename U>
        requires std::constructible_from<typename Base::storage_type, U &&>
    Broadcast(U &&child, const extents_type &extents)
      : Base(std::forward<U>(child)), m_extents(extents) {
        [&]<rank_type... K>(std::integer_sequence<rank_type, K...>) {
            ZIPPER_ASSERT(((expression().extent(K)
                            == m_extents.extent(detail_type::axis_map::reduced_rank_to_full_indices[K]))
                           && ...));
        }(std::make_integer_sequence<rank_type, detail_type::child_traits::extents_type::rank()>{});
    }

    constexpr auto extent(rank_type i) const -> index_type { return m_extents.extent(i); }
    constexpr auto extents() const -> extents_type { return m_extents; }

    template <typename... Args>
    auto coeff(Args &&...args) const -> value_type {
        return [&]<rank_type... K>(std::integer_sequence<rank_type, K...>) -> value_type {
            return expression()(zipper::detail::pack_index<
                                detail_type::axis_map::reduced_rank_to_full_indices[K]>(args...)...);
        }(std::make_integer_sequence<rank_type, detail_type::child_traits::extents_type::rank()>{});
    }

    auto make_owned() const {
        auto owned_child = expression().make_owned();
        return Broadcast<const decltype(owned_child), Extents, Axes...>(std::move(owned_child),
                                                                        m_extents);
    }

  private:
    extents_type m_extents;
};
} // namespace unary
} // namespace zipper::expression

#endif
