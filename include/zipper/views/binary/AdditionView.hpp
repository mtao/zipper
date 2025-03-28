
#if !defined(ZIPPER_VIEWS_BINARY_ADDITIONVIEW_HPP)
#define ZIPPER_VIEWS_BINARY_ADDITIONVIEW_HPP

#include "BinaryViewBase.hpp"
#include "detail/CoeffWiseTraits.hpp"
#include "zipper/concepts/ViewDerived.hpp"

namespace zipper::views {
namespace binary {
template <concepts::ViewDerived A, concepts::ViewDerived B>
class AdditionView;

}
template <typename A, typename B>
struct detail::ViewTraits<binary::AdditionView<A, B>>
    : public binary::detail::DefaultBinaryViewTraits<A, B> {
    using ATraits = detail::ViewTraits<A>;
    using BTraits = detail::ViewTraits<B>;
    using ConvertExtentsUtil = binary::detail::coeffwise_extents_values<
        typename ATraits::extents_type, typename BTraits::extents_type>;
    using extents_type = typename ConvertExtentsUtil::merged_extents_type;
};

namespace binary {
template <concepts::ViewDerived A, concepts::ViewDerived B>
class AdditionView : public BinaryViewBase<AdditionView<A, B>, A, B> {
   public:
    using self_type = AdditionView<A, B>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using extents_type = typename traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    using Base = BinaryViewBase<self_type, A, B>;

    using Base::Base;
    using Base::lhs;
    using Base::rhs;
    AdditionView(const A& a, const B& b)
        requires(!extents_traits::is_static)
        : Base(a, b, a.extents()) {}

    template <typename... Args>
    auto coeff(Args&&... idxs) const
        requires((std::is_convertible_v<Args, index_type> && ...))
    {
        static_assert(sizeof...(idxs) == extents_type::rank());
        static_assert(A::extents_type::rank() == extents_type::rank());
        static_assert(B::extents_type::rank() == extents_type::rank());
        return lhs()(idxs...) + rhs()(idxs...);
    }

};  // namespace binarytemplate<typenameA,typenameB>class AdditionView

template <concepts::ViewDerived A, concepts::ViewDerived B>
AdditionView(const A& a, const B& b) -> AdditionView<A, B>;
}  // namespace binary
}  // namespace zipper::views
#endif
