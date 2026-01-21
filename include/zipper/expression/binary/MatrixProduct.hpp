#if !defined(ZIPPER_VIEWS_BINARY_MATRIXPRODUCTVIEW_HPP)
#define ZIPPER_VIEWS_BINARY_MATRIXPRODUCTVIEW_HPP

#include "BinaryViewBase.hpp"
#include "zipper/concepts/MatrixViewDerived.hpp"
#include "zipper/views/DimensionedViewBase.hpp"
#include "zipper/views/detail/intersect_nonzeros.hpp"

namespace zipper::views {
namespace binary {
template <zipper::concepts::MatrixViewDerived A, zipper::concepts::MatrixViewDerived B>
class MatrixProductView;

}
namespace detail {

template <typename, typename>
struct coeffwise_extents_values;
template <index_type AR, index_type AC, index_type BR, index_type BC>
struct coeffwise_extents_values<extents<AR, AC>, extents<BR, BC>> {
    using product_extents_type = zipper::extents<AR, BC>;

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
template <zipper::concepts::MatrixViewDerived A, zipper::concepts::MatrixViewDerived B>
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
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace binary {
template <zipper::concepts::MatrixViewDerived A, zipper::concepts::MatrixViewDerived B>
class MatrixProductView : public BinaryViewBase<MatrixProductView<A, B>, A, B> {
   public:
    using self_type = MatrixProductView<A, B>;
    using ViewBase<self_type>::operator();
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using value_type = traits::value_type;
    using Base = BinaryViewBase<self_type, A, B>;
    using Base::extent;
    using Base::lhs;
    using Base::rhs;
    using lhs_traits = traits::ATraits;
    using rhs_traits = traits::BTraits;

    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    MatrixProductView(const A& a, const B& b)
        : Base(a, b,
               traits::ConvertExtentsUtil::merge(a.extents(), b.extents())) {
        assert(a.extent(1) == b.extent(0));
    }

    value_type coeff(index_type a, index_type b) const {
        value_type v = 0;
        constexpr bool lhs_sparse = lhs_traits::is_sparse(1);
        constexpr bool rhs_sparse = rhs_traits::is_sparse(0);
        if constexpr (lhs_sparse && rhs_sparse) {
            const auto& lnnz = lhs().template nonZeros<1>(a, b);
            const auto& rnnz = rhs().template nonZeros<0>(a, b);
            auto nnz = views::detail::intersect_nonzeros(lnnz, rnnz);

            for (const auto& j : nnz) {
                v += lhs()(a, j) * rhs()(j, b);
            }
        } else if constexpr (lhs_sparse) {
            const auto& lnnz = lhs().template nonZeros<1>(a, b);
            for (const auto& j : lnnz) {
                v += lhs()(a, j) * rhs()(j, b);
            }
        } else if constexpr (rhs_sparse) {
            const auto& rnnz = rhs().template nonZeros<0>(a, b);
            for (const auto& j : rnnz) {
                v += lhs()(a, j) * rhs()(j, b);
            }

        } else {
            for (index_type j = 0; j < lhs().extent(1); ++j) {
                v += lhs()(a, j) * rhs()(j, b);
            }
        }
        return v;
    }

};  // namespace binarytemplate<typenameA,typenameB>class MatrixProductView

template <zipper::concepts::MatrixViewDerived A, zipper::concepts::MatrixViewDerived B>
MatrixProductView(const A& a, const B& b) -> MatrixProductView<A, B>;
}  // namespace binary
}  // namespace zipper::views
#endif
