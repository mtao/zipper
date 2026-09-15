#if !defined(ZIPPER_expression_UNARY_SCALAROPERATIONVIEW_HPP)
#define ZIPPER_expression_UNARY_SCALAROPERATIONVIEW_HPP

#include "UnaryExpressionBase.hpp"
#include "concepts/ScalarOperation.hpp"
#include "detail/ZeroPreserving.hpp"
#include "zipper/expression/detail/IndexSet.hpp"

namespace zipper::expression {
namespace unary {
    namespace detail {
        template <zipper::concepts::QualifiedExpression Child, typename Scalar,
                  bool ScalarOnRight>
        struct scalar_operation_types {
            using child_value_type =
                typename DefaultUnaryExpressionDetail<Child>::base_value_type;
            using left_type = std::conditional_t<ScalarOnRight, child_value_type, Scalar>;
            using right_type = std::conditional_t<ScalarOnRight, Scalar, child_value_type>;

            template <typename Operation>
            static constexpr bool callable =
                concepts::ScalarOperation<left_type, Operation, right_type>;
        };
    } // namespace detail

    template <zipper::concepts::QualifiedExpression Child,
              typename Operation,
              typename Scalar,
              bool ScalarOnRight = false>
        requires detail::scalar_operation_types<Child, Scalar, ScalarOnRight>::
            template callable<Operation>
    class ScalarOperation;

}
template <zipper::concepts::QualifiedExpression Child,
          typename Operation,
          typename Scalar,
          bool ScalarOnRight>
struct detail::ExpressionTraits<
    unary::ScalarOperation<Child, Operation, Scalar, ScalarOnRight>>
  : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
        Child,
        zipper::detail::AccessFeatures::const_value()> {
    using ChildTraits = ExpressionTraits<std::decay_t<Child>>;
    using operand_types = unary::detail::scalar_operation_types<Child, Scalar, ScalarOnRight>;
    using left_type = typename operand_types::left_type;
    using right_type = typename operand_types::right_type;
    using value_type = std::decay_t<decltype(std::declval<const Operation &>()(
        std::declval<const left_type &>(), std::declval<const right_type &>()))>;

    /// Propagate has_index_set when the scalar Op preserves zeros.
    constexpr static bool has_index_set =
        ChildTraits::has_index_set
        && zipper::expression::detail::ZeroPreservingScalarOp<Operation,
                                                              ScalarOnRight>;
};

namespace unary {
    template <zipper::concepts::QualifiedExpression Child,
              typename Operation,
              typename Scalar,
              bool ScalarOnRight>
        requires detail::scalar_operation_types<Child, Scalar, ScalarOnRight>::
            template callable<Operation>
    class ScalarOperation
      : public UnaryExpressionBase<
            ScalarOperation<Child, Operation, Scalar, ScalarOnRight>,
            Child> {
      public:
        using self_type =
            ScalarOperation<Child, Operation, Scalar, ScalarOnRight>;
        using traits = zipper::expression::detail::ExpressionTraits<self_type>;
        using extents_type = traits::extents_type;
        using value_type = traits::value_type;

        using Base = UnaryExpressionBase<self_type, Child>;
        using Base::expression;

      private:
        template <typename U>
        static constexpr bool can_construct =
            std::constructible_from<typename Base::storage_type, U &&>
            && std::constructible_from<Operation, const Operation &>
            && std::constructible_from<Scalar, const Scalar &>;

      public:
        template <typename U>
            requires(ScalarOnRight && can_construct<U>)
        ScalarOperation(U &&a, const Scalar &b, const Operation &op)
          : Base(std::forward<U>(a)), m_op(op), m_scalar(b) {}
        template <typename U>
            requires(!ScalarOnRight && can_construct<U>)
        ScalarOperation(const Scalar &a, U &&b, const Operation &op)
          : Base(std::forward<U>(b)), m_op(op), m_scalar(a) {}

        template <typename U>
            requires(ScalarOnRight && can_construct<U>
                     && std::default_initializable<Operation>)
        ScalarOperation(U &&a, const Scalar &b)
          : ScalarOperation(std::forward<U>(a), b, Operation{}) {}

        template <typename U>
            requires(!ScalarOnRight && can_construct<U>
                     && std::default_initializable<Operation>)
        ScalarOperation(const Scalar &a, U &&b)
          : ScalarOperation(a, std::forward<U>(b), Operation{}) {}

        using child_value_type = traits::base_value_type;

        auto get_value(const child_value_type &value) const -> value_type {
            if constexpr (ScalarOnRight) {
                return value_type(m_op(value, m_scalar));
            } else {
                return value_type(m_op(m_scalar, value));
            }
        }

        /// Recursively deep-copy child so the result owns all data.
        auto make_owned() const {
            auto owned_child = expression().make_owned();
            if constexpr (ScalarOnRight) {
                return ScalarOperation<const decltype(owned_child),
                                       Operation,
                                       Scalar,
                                       ScalarOnRight>(
                    std::move(owned_child), m_scalar, m_op);
            } else {
                return ScalarOperation<const decltype(owned_child),
                                       Operation,
                                       Scalar,
                                       ScalarOnRight>(
                    m_scalar, std::move(owned_child), m_op);
            }
        }

        // ── Index set forwarding ────────────────────────────────────────────
        // Zero-preserving scalar ops (e.g. multiplies, divides-on-right)
        // don't change the sparsity pattern.

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
        Scalar m_scalar;
    };

    template <zipper::concepts::Expression Child,
              typename Operation,
              typename Scalar>
    ScalarOperation(const Child &a, const Scalar &b, const Operation &op)
        -> ScalarOperation<const Child &, Operation, Scalar, true>;
    template <zipper::concepts::Expression Child,
              typename Operation,
              typename Scalar>
    ScalarOperation(const Scalar &a, const Child &b, const Operation &op)
        -> ScalarOperation<const Child &, Operation, Scalar, false>;
} // namespace unary
} // namespace zipper::expression
#endif
