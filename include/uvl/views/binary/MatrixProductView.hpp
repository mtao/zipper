#if !defined(UVL_VIEWS_BINARY_MATRIXPRODUCTVIEW_HPP)
#define UVL_VIEWS_BINARY_MATRIXPRODUCTVIEW_HPP

#include "BinaryViewBase.hpp"
#include "uvl/concepts/MatrixViewDerived.hpp"
#include "uvl/views/DimensionedViewBase.hpp"

namespace uvl::views {
namespace binary {
template <concepts::MatrixViewDerived A, concepts::MatrixViewDerived B>
class MatrixProductView;

}
namespace detail {

template <typename, typename>
struct coeffwise_extents_values;
template <index_type AR, index_type AC, index_type BR, index_type BC>
struct coeffwise_extents_values<extents<AR, AC>, extents<BR, BC>> {
    using product_extents_type = uvl::extents<AR, BC>;

    constexpr static product_extents_type merge(const extents<AR, AC>& a,
                                                const extents<BR, BC>& b) {
        if constexpr (AR == std::dynamic_extent && BC == std::dynamic_extent) {
            return product_extents_type(a.extent(0), b.extent(1));
        } else if constexpr (AR == std::dynamic_extent) {
            return product_extents_type(a.extent(0));
        } else if constexpr (BC == std::dynamic_extent) {
            return product_extents_type(b.extent(1));
        } else {
            return {};
        }
    }
};
}  // namespace detail
template <typename A, typename B>
struct detail::ViewTraits<binary::MatrixProductView<A, B>>
    : public binary::detail::DefaultBinaryViewTraits<A, B>
//: public binary::detail::MatrixWiseTraits<A, B> {
//: public detail::ViewTraits<A> {
{
    using ATraits = detail::ViewTraits<A>;
    using BTraits = detail::ViewTraits<B>;
    using ConvertExtentsUtil =
        coeffwise_extents_values<typename ATraits::extents_type,
                                 typename BTraits::extents_type>;
    using extents_type = typename ConvertExtentsUtil::product_extents_type;
};

namespace binary {
template <concepts::MatrixViewDerived A, concepts::MatrixViewDerived B>
class MatrixProductView : public BinaryViewBase<MatrixProductView<A, B>, A, B> {
   public:
    using self_type = MatrixProductView<A, B>;
    using ViewBase<self_type>::operator();
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using value_type = traits::value_type;
    using Base = BinaryViewBase<self_type, A, B>;
    using Base::extent;
    using Base::lhs;
    using Base::rhs;

    using extents_type = traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;

    MatrixProductView(const A& a, const B& b)
        requires(extents_traits::is_static)
        : Base(a, b) {
        assert(a.extent(1) == b.extent(0));
    }
    MatrixProductView(const A& a, const B& b)
        requires(!extents_traits::is_static)
        : Base(a, b,
               traits::ConvertExtentsUtil::merge(a.extents(), b.extents())) {}

    value_type coeff(index_type a, index_type b) const {
        value_type v = 0;
        for (index_type j = 0; j < lhs().extent(1); ++j) {
            v += lhs()(a, j) * rhs()(j, b);
        }
        return v;
    }

};  // namespace binarytemplate<typenameA,typenameB>class MatrixProductView

template <concepts::MatrixViewDerived A, concepts::MatrixViewDerived B>
MatrixProductView(const A& a, const B& b) -> MatrixProductView<A, B>;
}  // namespace binary
}  // namespace uvl::views
#endif
