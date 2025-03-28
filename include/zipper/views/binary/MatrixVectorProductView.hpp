
#if !defined(ZIPPER_VIEWS_BINARY_MATRIXVECTORPRODUCTVIEW_HPP)
#define ZIPPER_VIEWS_BINARY_MATRIXVECTORPRODUCTVIEW_HPP

#include "BinaryViewBase.hpp"
#include "zipper/concepts/MatrixViewDerived.hpp"
#include "zipper/concepts/VectorViewDerived.hpp"

namespace zipper::views {
namespace binary {
template <concepts::MatrixViewDerived A, concepts::VectorViewDerived B>
class MatrixVectorProductView;

}
template <typename A, typename B>
struct detail::ViewTraits<binary::MatrixVectorProductView<A, B>>
    : public binary::detail::DefaultBinaryViewTraits<A, B> {
    using ATraits = detail::ViewTraits<A>;
    using BTraits = detail::ViewTraits<B>;
    using extents_type = extents<ATraits::extents_type::static_extent(0)>;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace binary {
template <concepts::MatrixViewDerived A, concepts::VectorViewDerived B>
class MatrixVectorProductView
    : public BinaryViewBase<MatrixVectorProductView<A, B>, A, B> {
   public:
    using self_type = MatrixVectorProductView<A, B>;
    using ViewBase<self_type>::operator();
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using value_type = traits::value_type;
    using Base = BinaryViewBase<self_type, A, B>;
    using extents_type = typename traits::extents_type;

    using Base::Base;
    using Base::lhs;
    using Base::rhs;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    MatrixVectorProductView(const A& a, const B& b)
        requires(extents_traits::is_dynamic)
        : Base(a, b, extents_type{a.extent(0)}) {
        assert(a.extent(1) == b.extent(0));
    }

    MatrixVectorProductView(const A& a, const B& b)
        requires(extents_traits::is_static)
        : Base(a, b) {
        assert(a.extent(1) == b.extent(0));
    }
    value_type coeff(index_type a) const {
        value_type v = 0;
        for (index_type j = 0; j < lhs().extent(1); ++j) {
            v += lhs()(a, j) * rhs()(j);
        }
        return v;
    }
};

template <concepts::MatrixViewDerived A, concepts::MatrixViewDerived B>
MatrixVectorProductView(const A& a, const B& b)
    -> MatrixVectorProductView<A, B>;
}  // namespace binary
}  // namespace zipper::views
#endif
