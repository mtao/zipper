#if !defined(UVL_VIEWS_UNARY_SCALARARITHMETICVIEW_HPP)
#define UVL_VIEWS_UNARY_SCALARARITHMETICVIEW_HPP

#include <functional>

#include "OperationView.hpp"
#include "ScalarOperationView.hpp"

namespace uvl::views::unary {

template <concepts::ViewDerived View>
using NegateView = OperationView<
    View,
    std::negate<typename uvl::views::detail::ViewTraits<View>::value_type> >;

template <concepts::ViewDerived View>
using LogicalNotView = OperationView<
    View, std::logical_not<
              typename uvl::views::detail::ViewTraits<View>::value_type> >;

template <concepts::ViewDerived View>
using BitNotView = OperationView<
    View,
    std::bit_not<typename uvl::views::detail::ViewTraits<View>::value_type> >;

template <typename Scalar, concepts::ViewDerived View, bool ScalarOnRight>
using ScalarPlusView =
    ScalarOperationView<View, std::plus<Scalar>, Scalar, ScalarOnRight>;

template <typename Scalar, concepts::ViewDerived View, bool ScalarOnRight>
using ScalarMinusView =
    ScalarOperationView<View, std::minus<Scalar>, Scalar, ScalarOnRight>;

template <typename Scalar, concepts::ViewDerived View, bool ScalarOnRight>
using ScalarMultipliesView =
    ScalarOperationView<View, std::multiplies<Scalar>, Scalar, ScalarOnRight>;

template <typename Scalar, concepts::ViewDerived View, bool ScalarOnRight>
using ScalarDividesView =
    ScalarOperationView<View, std::divides<Scalar>, Scalar, ScalarOnRight>;

template <typename Scalar, concepts::ViewDerived View, bool ScalarOnRight>
using ScalarModulusView =
    ScalarOperationView<View, std::modulus<Scalar>, Scalar, ScalarOnRight>;

template <typename Scalar, concepts::ViewDerived View, bool ScalarOnRight>
using ScalarEqualsToView =
    ScalarOperationView<View, std::equal_to<Scalar>, Scalar, ScalarOnRight>;

template <typename Scalar, concepts::ViewDerived View, bool ScalarOnRight>
using ScalarNotEqualsToView =
    ScalarOperationView<View, std::not_equal_to<Scalar>, Scalar, ScalarOnRight>;

template <typename Scalar, concepts::ViewDerived View, bool ScalarOnRight>
using ScalarGreaterView =
    ScalarOperationView<View, std::greater<Scalar>, Scalar, ScalarOnRight>;

template <typename Scalar, concepts::ViewDerived View, bool ScalarOnRight>
using ScalarLessView =
    ScalarOperationView<View, std::less<Scalar>, Scalar, ScalarOnRight>;

template <typename Scalar, concepts::ViewDerived View, bool ScalarOnRight>
using ScalarGreaterEqualView =
    ScalarOperationView<View, std::greater_equal<Scalar>, Scalar,
                        ScalarOnRight>;

template <typename Scalar, concepts::ViewDerived View, bool ScalarOnRight>
using ScalarLessEqualView =
    ScalarOperationView<View, std::less_equal<Scalar>, Scalar, ScalarOnRight>;

template <typename Scalar, concepts::ViewDerived View, bool ScalarOnRight>
using ScalarLogicalAndView =
    ScalarOperationView<View, std::logical_and<Scalar>, Scalar, ScalarOnRight>;

template <typename Scalar, concepts::ViewDerived View, bool ScalarOnRight>
using ScalarLogicalOrView =
    ScalarOperationView<View, std::logical_or<Scalar>, Scalar, ScalarOnRight>;

template <typename Scalar, concepts::ViewDerived View, bool ScalarOnRight>
using ScalarBitAndView =
    ScalarOperationView<View, std::bit_and<Scalar>, Scalar, ScalarOnRight>;

template <typename Scalar, concepts::ViewDerived View, bool ScalarOnRight>
using ScalarBitOrView =
    ScalarOperationView<View, std::bit_or<Scalar>, Scalar, ScalarOnRight>;

template <typename Scalar, concepts::ViewDerived View, bool ScalarOnRight>
using ScalarBitXorView =
    ScalarOperationView<View, std::bit_xor<Scalar>, Scalar, ScalarOnRight>;

}  // namespace uvl::views::unary

#endif
