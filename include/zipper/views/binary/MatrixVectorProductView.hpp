#if !defined(ZIPPER_VIEWS_BINARY_MATRIXVECTORPRODUCTVIEW_HPP)
#define ZIPPER_VIEWS_BINARY_MATRIXVECTORPRODUCTVIEW_HPP

#include "BinaryViewBase.hpp"
#include "zipper/concepts/MatrixViewDerived.hpp"
#include "zipper/concepts/VectorViewDerived.hpp"
#include "zipper/views/detail/intersect_nonzeros.hpp"

namespace zipper::views {
namespace binary {
template <zipper::concepts::MatrixViewDerived A, zipper::concepts::VectorViewDerived B>
class MatrixVectorProductView;

}
template <zipper::concepts::MatrixViewDerived A, zipper::concepts::VectorViewDerived B>
struct detail::ViewTraits<binary::MatrixVectorProductView<A, B>>
    : public binary::detail::DefaultBinaryViewTraits<A, B> {
    using ATraits = detail::ViewTraits<A>;
    using BTraits = detail::ViewTraits<B>;
    using extents_type = extents<ATraits::extents_type::static_extent(0)>;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace binary {
template <zipper::concepts::MatrixViewDerived A, zipper::concepts::VectorViewDerived B>
class MatrixVectorProductView
    : public BinaryViewBase<MatrixVectorProductView<A, B>, A, B> {
   public:
    using self_type = MatrixVectorProductView<A, B>;
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
        constexpr bool lhs_sparse = lhs_traits::is_sparse(1);
        constexpr bool rhs_sparse = rhs_traits::is_sparse(0);
        if constexpr (lhs_sparse && rhs_sparse) {
            const auto& lnnz = lhs().template nonZeros<1>(a);
            const auto& rnnz = rhs().template nonZeros<0>(a);
            auto nnz = views::detail::intersect_nonzeros(lnnz, rnnz);

            for (const auto& j : nnz) {
                v += lhs()(a, j) * rhs()(j);
            }
        } else if constexpr (lhs_sparse) {
            const auto& lnnz = lhs().template nonZeros<1>(a);
            for (const auto& j : lnnz) {
                v += lhs()(a, j) * rhs()(j);
            }
        } else if constexpr (rhs_sparse) {
            const auto& rnnz = rhs().template nonZeros<0>(a);
            for (const auto& j : rnnz) {
                v += lhs()(a, j) * rhs()(j);
            }

        } else {
            for (index_type j = 0; j < lhs().extent(1); ++j) {
                v += lhs()(a, j) * rhs()(j);
            }
        }
        return v;
    }
};

template <zipper::concepts::MatrixViewDerived A, zipper::concepts::MatrixViewDerived B>
MatrixVectorProductView(const A& a, const B& b)
    -> MatrixVectorProductView<A, B>;
}  // namespace binary
}  // namespace zipper::views
#endif
