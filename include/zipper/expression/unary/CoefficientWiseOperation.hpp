#if !defined(ZIPPER_EXPRESSION_UNARY_COEFFICIENTWISE_OPERATION_HPP)
#define ZIPPER_EXPRESSION_UNARY_COEFFICIENTWISE_OPERATION_HPP

#include "UnaryExpressionBase.hpp"
#include "ZeroPreserving.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/expression/detail/IndexSet.hpp"

namespace zipper::expression {
namespace unary {
    template <zipper::concepts::QualifiedExpression Child, typename Op>
    class CoefficientWiseOperation;

} // namespace unary
template <zipper::concepts::QualifiedExpression Child, typename Op>
struct expression::detail::ExpressionTraits<
    unary::CoefficientWiseOperation<Child, Op>>
  : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
        Child,
        zipper::detail::AccessFeatures{.is_const = true,
                                       .is_reference = false}> {
    using child_traits = ExpressionTraits<std::decay_t<Child>>;
    using value_type = std::decay_t<decltype(std::declval<Op>()(
        std::declval<typename child_traits::value_type>()))>;

    /// Propagate has_index_set when the Op preserves zeros.
    constexpr static bool has_index_set =
        child_traits::has_index_set
        && zipper::expression::detail::ZeroPreservingUnaryOp<Op>;

    /// Backward-compatible alias for has_index_set.
    constexpr static bool has_known_zeros = has_index_set;
};

// represents a coefficient-wise transformation of an underlyng expression
namespace unary {

    template <zipper::concepts::QualifiedExpression Child, typename Operation>
    class CoefficientWiseOperation
      : public UnaryExpressionBase<CoefficientWiseOperation<Child, Operation>,
                                   Child> {
      public:
        using self_type = CoefficientWiseOperation<Child, Operation>;
        using Base = UnaryExpressionBase<self_type, Child>;
        using traits = zipper::expression::detail::ExpressionTraits<self_type>;

        using extents_type = traits::extents_type;
        using value_type = traits::value_type;

        using Base::expression;
        using Base::extent;

        template <typename U>
            requires std::constructible_from<typename Base::storage_type, U &&>
        CoefficientWiseOperation(U &&v, Operation const &op = {})
          : Base(std::forward<U>(v)), m_op(op) {}

        using child_value_type = traits::base_value_type;

        auto get_value(const child_value_type &value) const -> value_type {
            return value_type(m_op(value));
        }

        /// Recursively deep-copy child so the result owns all data.
        auto make_owned() const {
            auto owned_child = expression().make_owned();
            return CoefficientWiseOperation<const decltype(owned_child),
                                            Operation>(std::move(owned_child),
                                                       m_op);
        }

        // ── Index set forwarding ────────────────────────────────────────────
        // Zero-preserving ops (e.g. negate) don't change the sparsity pattern.

        template <rank_type D, typename... Args>
            requires(traits::has_index_set)
        auto index_set(Args &&...args) const {
            return expression().template index_set<D>(
                std::forward<Args>(args)...);
        }

        /// @deprecated Use index_set instead.
        template <rank_type D, typename... Args>
            requires(traits::has_index_set)
        auto nonzero_range(Args &&...args) const {
            return index_set<D>(std::forward<Args>(args)...);
        }

        auto col_range_for_row(index_type row) const
            requires(traits::has_index_set && extents_type::rank() == 2)
        {
            return expression().col_range_for_row(row);
        }

        auto row_range_for_col(index_type col) const
            requires(traits::has_index_set && extents_type::rank() == 2)
        {
            return expression().row_range_for_col(col);
        }

        auto nonzero_segment() const
            requires(traits::has_index_set && extents_type::rank() == 1)
        {
            return expression().nonzero_segment();
        }

      private:
        Operation m_op;
    };

    template <typename Op, zipper::concepts::Expression B>
    CoefficientWiseOperation(const B &a, const Op &)
        -> CoefficientWiseOperation<const B &, Op>;
} // namespace unary
} // namespace zipper::expression
#endif
