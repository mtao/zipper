
#if !defined(ZIPPER_VIEWS_BINARY_CROSSPRODUCTVIEW_HPP)
#define ZIPPER_VIEWS_BINARY_CROSSPRODUCTVIEW_HPP

#include "BinaryViewBase.hpp"
#include "detail/CoeffWiseTraits.hpp"
#include "zipper/concepts/MatrixViewDerived.hpp"
#include "zipper/concepts/VectorViewDerived.hpp"
#include "zipper/views/detail/intersect_nonzeros.hpp"

namespace zipper::views {
namespace binary {
template <zipper::concepts::VectorViewDerived A, zipper::concepts::VectorViewDerived B>
class CrossProductView;

}
template <typename A, typename B>
struct detail::ViewTraits<binary::CrossProductView<A, B>>
    : public binary::detail::DefaultBinaryViewTraits<A, B> {
    using ATraits = detail::ViewTraits<A>;
    using BTraits = detail::ViewTraits<B>;
    using ConvertExtentsUtil = binary::detail::coeffwise_extents_values<
        typename ATraits::extents_type, typename BTraits::extents_type>;
    using extents_type = typename ConvertExtentsUtil::merged_extents_type;

    // constraint on the extent, supporting 2d is ugly but who cares
    static_assert(extents_type::static_extent(0) == std::dynamic_extent ||
                  extents_type::static_extent(0) == 3);
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace binary {
template <zipper::concepts::VectorViewDerived A, zipper::concepts::VectorViewDerived B>
class CrossProductView : public BinaryViewBase<CrossProductView<A, B>, A, B> {
   public:
    using self_type = CrossProductView<A, B>;
    using ViewBase<self_type>::operator();
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using value_type = traits::value_type;
    using Base = BinaryViewBase<self_type, A, B>;
    using extents_type = typename traits::extents_type;
    using lhs_traits = traits::ATraits;
    using rhs_traits = traits::BTraits;

    using Base::Base;
    using Base::lhs;
    using Base::rhs;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    CrossProductView(const A& a, const B& b)
        requires(extents_traits::is_dynamic)
        : Base(a, b, extents_type{a.extent(0)}) {
        assert(a.extent(0) == b.extent(0));
        assert(a.extent(0) == 3);
    }

    CrossProductView(const A& a, const B& b)
        requires(extents_traits::is_static)
        : Base(a, b) {
        assert(a.extent(0) == b.extent(0));
    }
    value_type coeff(index_type a) const {
        rank_type b = (a + 1) % 3;
        rank_type c = (a + 2) % 3;
        return lhs()(b) * rhs()(c) - lhs()(c) * rhs()(b);
    }
};

template <zipper::concepts::VectorViewDerived A, zipper::concepts::VectorViewDerived B>
CrossProductView(const A& a, const B& b) -> CrossProductView<A, B>;
}  // namespace binary
}  // namespace zipper::views
#endif
