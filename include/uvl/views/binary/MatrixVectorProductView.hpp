
#if !defined(UVL_VIEWS_BINARY_MATRIXVECTORPRODUCTVIEW_HPP)
#define UVL_VIEWS_BINARY_MATRIXVECTORPRODUCTVIEW_HPP

#include "uvl/concepts/MatrixViewDerived.hpp"
#include "uvl/concepts/VectorViewDerived.hpp"
#include "uvl/views/DimensionedViewBase.hpp"

namespace uvl::views {
namespace binary {
template <concepts::MatrixViewDerived A, concepts::VectorViewDerived B>
class MatrixVectorProductView;

}
template <typename A, typename B>
struct detail::ViewTraits<binary::MatrixVectorProductView<A, B>>
//: public binary::detail::MatrixWiseTraits<A, B> {
//: public detail::ViewTraits<A> {
{
    using ATraits = detail::ViewTraits<A>;
    using BTraits = detail::ViewTraits<A>;
    // static_assert(std::is_same_v<ATraits::value_type,BTraits::value_type>);
    using extents_type =
        extents<typename ATraits::extents_type::static_extent(0)>;
    using value_type = typename ATraits::value_type;
    constexpr static bool is_writable = false;
};

namespace binary {
template <concepts::MatrixViewDerived A, concepts::VectorViewDerived B>
class MatrixVectorProductView
    : public DimensionedViewBase<MatrixVectorProductView<A, B>> {
   public:
    using self_type = MatrixVectorProductView<A, B>;
    using ViewBase<self_type>::operator();
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using value_type = traits::value_type;
    using Base = DimensionedViewBase<self_type>;
    using Base::extent;
    using extents_type = traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;

    MatrixVectorProductView(const A& a, const B& b)
        requires(traits::ATraits::extents_type::static_rank(0) ==
                 std::dynamic_extent)
        : m_lhs(a), m_rhs(b) {
        assert(a.extent(1) == b.extent(0));
    }

    MatrixVectorProductView(const A& a, const B& b)
        requires(traits::ATraits::extents_type::static_rank(0) !=
                 std::dynamic_extent)
        : m_lhs(a), m_rhs(b), m_extents(a.extent(0)) {
        assert(a.extent(1) == b.extent(0));
    }
    const extents_type& extents() const { return m_extents; }

    // using value_type = traits::value_type;

    // const mapping_type& mapping() const { return derived().mapping(); }
    // const extents_type& extents() const { return derived().extents(); }
    value_type coeff(index_type a) const {
        value_type v = 0;
        for (index_type j = 0; j < m_lhs.extent(1); ++j) {
            v += m_lhs(a, j) * m_rhs(j);
        }
        return v;
    }

   private:
    const A& m_lhs;
    const B& m_rhs;
    extents_type m_extents;

};  // namespace binarytemplate<typenameA,typenameB>class
    // MatrixVectorProductView

template <concepts::MatrixViewDerived A, concepts::MatrixViewDerived B>
MatrixVectorProductView(const A& a, const B& b)
    -> MatrixVectorProductView<A, B>;
}  // namespace binary
}  // namespace uvl::views
#endif
