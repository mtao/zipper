#if !defined(UVL_VIEWS_UNARY_SCALAROPERATIONVIEW_HPP)
#define UVL_VIEWS_UNARY_SCALAROPERATIONVIEW_HPP

#include "UnaryViewBase.hpp"

namespace uvl::views {
namespace unary {
template <concepts::ViewDerived Child, typename Operation, typename Scalar,
          bool ScalarOnRight = false>
class ScalarOperationView;

}
template <concepts::ViewDerived Child, typename Operation, typename Scalar,
          bool ScalarOnRight>
struct detail::ViewTraits<
    unary::ScalarOperationView<Child, Operation, Scalar, ScalarOnRight>>
    : public uvl::views::unary::detail::DefaultUnaryViewTraits<Child> {
    using ChildTraits = ViewTraits<Child>;
    using value_type = decltype(std::declval<Operation>()(
        std::declval<typename ChildTraits::value_type>(),
        std::declval<Scalar>()));
};

namespace unary {
template <concepts::ViewDerived Child, typename Operation, typename Scalar,
          bool ScalarOnRight>
class ScalarOperationView
    : public UnaryViewBase<
          ScalarOperationView<Child, Operation, Scalar, ScalarOnRight>, Child> {
   public:
    using self_type =
        ScalarOperationView<Child, Operation, Scalar, ScalarOnRight>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
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

template <concepts::ViewDerived Child, typename Operation, typename Scalar>
ScalarOperationView(const Child& a, const Scalar& b, const Operation& op)
    -> ScalarOperationView<Child, Scalar, Operation, true>;
template <concepts::ViewDerived Child, typename Operation, typename Scalar>
ScalarOperationView(const Scalar& a, const Child& b, const Operation& op)
    -> ScalarOperationView<Child, Scalar, Operation, false>;
}  // namespace unary
}  // namespace uvl::views
#endif
