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
    using extents_type = typename coeffwise_extents_values<
        typename ATraits::extents_type,
        typename BTraits::extents_type>::product_extents_type;
    using value_type = typename ATraits::value_type;
    constexpr static bool is_writable = false;
};

namespace binary {
template <concepts::MatrixViewDerived A, concepts::MatrixViewDerived B>
class MatrixProductView : public DimensionedViewBase<MatrixProductView<A, B>> {
   public:
    using self_type = MatrixProductView<A, B>;
    using ViewBase<self_type>::operator();
    MatrixProductView(const A& a, const B& b) : m_lhs(a), m_rhs(b) {
        assert(a.extent(1) == b.extent(0));
    }
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using value_type = traits::value_type;
    using Base = DimensionedViewBase<self_type>;
    using Base::extent;
    using Base::extents;

    // using value_type = traits::value_type;
    //  using extents_type = traits::extents_type;
    //  using extents_traits = uvl::detail::ExtentsTraits<extents_type>;

    // const mapping_type& mapping() const { return derived().mapping(); }
    // const extents_type& extents() const { return derived().extents(); }
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
};  // namespace binarytemplate<typenameA,typenameB>class MatrixProductView

template <concepts::MatrixViewDerived A, concepts::MatrixViewDerived B>
MatrixProductView(const A& a, const B& b) -> MatrixProductView<A, B>;
}  // namespace binary
}  // namespace uvl::views
#endif
