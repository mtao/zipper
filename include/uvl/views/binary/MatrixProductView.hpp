#if !defined(UVL_VIEWS_BINARY_MATRIXPRODUCTVIEW_HPP)
#define UVL_VIEWS_BINARY_MATRIXPRODUCTVIEW_HPP

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

    product_extents_type merge(const extents<AR,AC>& a, const extents<BR,BC>& b) {
        if constexpr(AR == std::dynamic_extent && BC == std::dynamic_extent) {
            return product_extents_type(a.extent(0),b.extent(1));
        } else 
        if constexpr(AR == std::dynamic_extent) {
            return product_extents_type(a.extent(0));
        } else if constexpr(BC == std::dynamic_extent) {
            return product_extents_type(b.extent(1));
        } else {
            return {};
        }
    }
};
}  // namespace detail
template <typename A, typename B>
struct detail::ViewTraits<binary::MatrixProductView<A, B>>
//: public binary::detail::MatrixWiseTraits<A, B> {
//: public detail::ViewTraits<A> {
{
    using ATraits = detail::ViewTraits<A>;
    using BTraits = detail::ViewTraits<A>;
    // static_assert(std::is_same_v<ATraits::value_type,BTraits::value_type>);
    using ConvertExtentsUtil = coeffwise_extents_values<
        typename ATraits::extents_type,
        typename BTraits::extents_type>;
    using extents_type = typename ConvertExtentsUtil::product_extents_type;
    using value_type = typename ATraits::value_type;
    constexpr static bool is_writable = false;
};

namespace binary {
template <concepts::MatrixViewDerived A, concepts::MatrixViewDerived B>
class MatrixProductView : public DimensionedViewBase<MatrixProductView<A, B>> {
   public:
    using self_type = MatrixProductView<A, B>;
    using ViewBase<self_type>::operator();
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using value_type = traits::value_type;
    using Base = DimensionedViewBase<self_type>;
    using Base::extent;

      using extents_type = traits::extents_type;
      using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
    
    MatrixProductView(const A& a, const B& b) requires(extents_traits::is_static): m_lhs(a), m_rhs(b) {
        assert(a.extent(1) == b.extent(0));
    }
    MatrixProductView(const A& a, const B& b) requires(!extents_traits::is_static)
        : m_lhs(a), m_rhs(b), m_extents(traits::ConvertExtentsUtil::merge(a.extents(),b.extents()))
    {}

    // const mapping_type& mapping() const { return derived().mapping(); }
     const extents_type& extents() const { return m_extents; }
    value_type coeff(index_type a, index_type b) const {
        value_type v = 0;
        for (index_type j = 0; j < m_lhs.extent(1); ++j) {
            v += m_lhs(a, j) * m_rhs(j, b);
        }
        return v;
    }

   private:
    const A& m_lhs;
    const B& m_rhs;
    extents_type m_extents;
};  // namespace binarytemplate<typenameA,typenameB>class MatrixProductView

template <concepts::MatrixViewDerived A, concepts::MatrixViewDerived B>
MatrixProductView(const A& a, const B& b) -> MatrixProductView<A, B>;
}  // namespace binary
}  // namespace uvl::views
#endif
