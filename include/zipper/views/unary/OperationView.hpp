#if !defined(ZIPPER_VIEWS_UNARY_OPERATIONVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_OPERATIONVIEW_HPP

#include "UnaryViewBase.hpp"

namespace zipper::views {
namespace unary {
// TODO: operation should inherit std::unary_function<T,U>
template <concepts::ViewDerived B, typename Operation>
class OperationView;

}  // namespace unary
template <concepts::ViewDerived Child, typename Operation>
struct detail::ViewTraits<unary::OperationView<Child, Operation>>
    : public zipper::views::unary::detail::DefaultUnaryViewTraits<Child> {
    using value_type = std::decay_t<decltype(std::declval<Operation>()(
        std::declval<typename Child::value_type>()))>;
};

namespace unary {
template <concepts::ViewDerived Child, typename Operation>
class OperationView
    : public UnaryViewBase<OperationView<Child, Operation>, Child> {
   public:
    using self_type = OperationView<Child, Operation>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;

    using Base = UnaryViewBase<self_type, Child>;
    using Base::extent;
    using Base::view;

    OperationView(const Child& v, const Operation& op = {})
        : Base(v), m_op(op) {}

    using child_value_type = traits::base_value_type;

    value_type get_value(const child_value_type& value) const {
        return m_op(value);
    }

   private:
    Operation m_op;
};

template <typename A, concepts::ViewDerived B>
OperationView(const A& a, const B& b) -> OperationView<A, B>;
}  // namespace unary
}  // namespace zipper::views
#endif
