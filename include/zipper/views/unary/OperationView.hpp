#if !defined(ZIPPER_VIEWS_UNARY_OPERATIONVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_OPERATIONVIEW_HPP

#include "UnaryViewBase.hpp"
#include "concepts/ScalarOperation.hpp"

namespace zipper::views {
namespace unary {
// TODO: operation should inherit std::unary_function<T,U>
template <zipper::concepts::ViewDerived B, 
         //
         //concepts::ScalarOperation<typename B::value_type> Operation,
         typename Operation,
         //
          bool PreservesZeros = false>
         //concepts::ScalarOperation<typename Child::value_type>
class OperationView;

}  // namespace unary
template <zipper::concepts::ViewDerived Child, 
         //
         //unary::concepts::ScalarOperation<typename Child::value_type> Operation,
         typename Operation,
         //
         bool PreservesZeros>
struct detail::ViewTraits<
    unary::OperationView<Child, Operation, PreservesZeros>>
    : public zipper::views::unary::detail::DefaultUnaryViewTraits<Child,
                                                                  false> {
    using value_type = std::decay_t<decltype(std::declval<Operation>()(
        std::declval<typename Child::value_type>()))>;
};

// represents a coefficient-wise transformation of an underlyng view
namespace unary {




template <zipper::concepts::ViewDerived Child, 
         //concepts::ScalarOperation<typename Child::value_type> Operation,
         typename Operation,
         //
         bool PreservesZeros>

         //requires(concepts::ScalarOperation<Child::value_type,Operation>)
class OperationView
    : public UnaryViewBase<OperationView<Child, Operation>, Child> {
   public:
    using self_type = OperationView<Child, Operation>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    constexpr static bool holds_extents = traits::holds_extents;
    static_assert(!holds_extents);
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;

    using Base = UnaryViewBase<self_type, Child>;
    using Base::extent;
    using Base::view;

    OperationView(const Child& v, const Operation& op = {})
        : Base(v), m_op(op) {}

    using child_value_type = traits::base_value_type;

    value_type get_value(const child_value_type& value) const {
        return value_type(m_op(value));
    }

   private:
    Operation m_op;
};

template <typename A, zipper::concepts::ViewDerived B>
OperationView(const A& a, const B& b) -> OperationView<A, B>;
}  // namespace unary
}  // namespace zipper::views
#endif
