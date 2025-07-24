

#if !defined(ZIPPER_VIEWS_BINARY_COEFFPRODUCTVIEW_HPP)
#define ZIPPER_VIEWS_BINARY_COEFFPRODUCTVIEW_HPP

#include "BinaryViewBase.hpp"
#include "detail/CoeffWiseTraits.hpp"
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/views/DimensionedViewBase.hpp"

namespace zipper::views {
namespace binary {
template <zipper::concepts::ViewDerived A, zipper::concepts::ViewDerived B>
class CoeffProductView;

}
template <typename A, typename B>
struct detail::ViewTraits<binary::CoeffProductView<A, B>>
    : public binary::detail::DefaultBinaryViewTraits<A, B> {
    using Base = detail::ViewTraits<A>;
    using extents_type = typename Base::extents_type;
};

namespace binary {
    // Per-coefficient product (i.e A(x,y,z) * B(x,y,z))
template <zipper::concepts::ViewDerived A, zipper::concepts::ViewDerived B>
class CoeffProductView : public BinaryViewBase<CoeffProductView<A, B>, A, B> {
   public:
    using self_type = CoeffProductView<A, B>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using Base = BinaryViewBase<self_type, A, B>;
    using Base::Base;
    using Base::extent;
    using Base::extents;
    using Base::lhs;
    using Base::rhs;
    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    CoeffProductView(const A& a, const B& b)
        requires(!extents_traits::is_static)
        : Base(a, b, a.extents()) {}

    template <typename... Args>
    auto coeff(Args&&... idxs) const {
        return lhs()(idxs...) * rhs()(idxs...);
    }

};  // namespace binarytemplate<typenameA,typenameB>class CoeffProductView

template <zipper::concepts::ViewDerived A, zipper::concepts::ViewDerived B>
CoeffProductView(const A& a, const B& b) -> CoeffProductView<A, B>;
}  // namespace binary
}  // namespace zipper::views
#endif
