#if !defined(ZIPPER_VIEWS_UNARY_SCALAROPERATIONVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_SCALAROPERATIONVIEW_HPP

#include "UnaryViewBase.hpp"

namespace zipper::views {
namespace unary {
template <zipper::concepts::QualifiedViewDerived Child, typename Operation, typename Scalar,
          bool ScalarOnRight = false, bool PreservesZeros=false>
class ScalarOperationView;

}
template <zipper::concepts::QualifiedViewDerived Child, typename Operation, typename Scalar,
          bool ScalarOnRight, bool PreservesZeros>
struct detail::ViewTraits<
    unary::ScalarOperationView<Child, Operation, Scalar, ScalarOnRight, PreservesZeros>>
    : public zipper::views::unary::detail::DefaultUnaryViewTraits<Child> {
    using ChildTraits = ViewTraits<Child>;
    using value_type = decltype(std::declval<Operation>()(
        std::declval<typename ChildTraits::value_type>(),
        std::declval<Scalar>()));
};

namespace unary {
template <zipper::concepts::QualifiedViewDerived Child, typename Operation, typename Scalar,
          bool ScalarOnRight, bool PreservesZeros>
class ScalarOperationView
    : public UnaryViewBase<
          ScalarOperationView<Child, Operation, Scalar, ScalarOnRight, PreservesZeros>, Child> {
   public:
    using self_type =
        ScalarOperationView<Child, Operation, Scalar, ScalarOnRight, PreservesZeros>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;

    using Base = UnaryViewBase<self_type, Child>;
    using Base::view;

    ScalarOperationView(const Child& a, const Scalar& b, const Operation& op = {})
        requires(ScalarOnRight)
        : Base(a), m_op(op), m_scalar(b) {}
    ScalarOperationView(const Scalar& a, const Child& b, const Operation& op = {})
        requires(!ScalarOnRight)
        : Base(b), m_op(op), m_scalar(a) {}

    // using child_value_type = traits::base_value_type;

    value_type get_value(const auto& value) const {
        if constexpr (ScalarOnRight) {
            return m_op(value, m_scalar);
        } else {
            return m_op(m_scalar, value);
        }
    }

   private:
    Operation m_op;
    Scalar m_scalar;
};

template <zipper::concepts::QualifiedViewDerived Child, typename Operation, typename Scalar>
ScalarOperationView(const Child& a, const Scalar& b, const Operation& op)
    -> ScalarOperationView<Child, Scalar, Operation, true>;
template <zipper::concepts::QualifiedViewDerived Child, typename Operation, typename Scalar>
ScalarOperationView(const Scalar& a, const Child& b, const Operation& op)
    -> ScalarOperationView<Child, Scalar, Operation, false>;
}  // namespace unary
}  // namespace zipper::views
#endif
