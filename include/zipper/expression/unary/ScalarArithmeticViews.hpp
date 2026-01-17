#if !defined(ZIPPER_VIEWS_UNARY_SCALARARITHMETICVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_SCALARARITHMETICVIEW_HPP

#include <functional>
#include <type_traits>

#include "OperationView.hpp"
#include "ScalarOperationView.hpp"

namespace zipper::views::unary {

template <zipper::concepts::QualifiedViewDerived View>
using NegateView = OperationView<
    View,
    std::negate<typename zipper::views::detail::ViewTraits<View>::value_type> >;

template <zipper::concepts::QualifiedViewDerived View>
using LogicalNotView = OperationView<
    View, std::logical_not<
              typename zipper::views::detail::ViewTraits<View>::value_type> >;

template <zipper::concepts::QualifiedViewDerived View>
using BitNotView =
    OperationView<View, std::bit_not<typename zipper::views::detail::ViewTraits<
                            View>::value_type> >;

template <typename Scalar, zipper::concepts::QualifiedViewDerived View,
          bool ScalarOnRight>
using ScalarPlusView =
    ScalarOperationView<View, std::plus<std::common_type_t<Scalar,typename View::value_type>>, Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::QualifiedViewDerived View,
          bool ScalarOnRight>
using ScalarMinusView =
    ScalarOperationView<View, std::minus<std::common_type_t<Scalar,typename View::value_type>>, Scalar, ScalarOnRight>;

// coeff-wise product preserves zero-ness
template <typename Scalar, zipper::concepts::QualifiedViewDerived View,
          bool ScalarOnRight>
using ScalarMultipliesView = ScalarOperationView<View, std::multiplies<std::common_type_t<Scalar,typename View::value_type>>,
                                                 Scalar, ScalarOnRight, true>;

template <typename Scalar, zipper::concepts::QualifiedViewDerived View,
          bool ScalarOnRight>
using ScalarDividesView =
    ScalarOperationView<View, std::divides<std::common_type_t<Scalar,typename View::value_type>>, Scalar, ScalarOnRight>;

// modulus preserves zero if the modulus is on the right
template <typename Scalar, zipper::concepts::QualifiedViewDerived View,
          bool ScalarOnRight>
using ScalarModulusView =
    ScalarOperationView<View, std::modulus<std::common_type_t<Scalar,typename View::value_type>>, Scalar, ScalarOnRight,
                        ScalarOnRight>;

template <typename Scalar, zipper::concepts::QualifiedViewDerived View,
          bool ScalarOnRight>
using ScalarEqualsToView =
    ScalarOperationView<View, std::equal_to<std::common_type_t<Scalar,typename View::value_type>>, Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::QualifiedViewDerived View,
          bool ScalarOnRight>
using ScalarNotEqualsToView =
    ScalarOperationView<View, std::not_equal_to<std::common_type_t<Scalar,typename View::value_type>>, Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::QualifiedViewDerived View,
          bool ScalarOnRight>
using ScalarGreaterView =
    ScalarOperationView<View, std::greater<std::common_type_t<Scalar,typename View::value_type>>, Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::QualifiedViewDerived View,
          bool ScalarOnRight>
using ScalarLessView =
    ScalarOperationView<View, std::less<std::common_type_t<Scalar,typename View::value_type>>, Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::QualifiedViewDerived View,
          bool ScalarOnRight>
using ScalarGreaterEqualView =
    ScalarOperationView<View, std::greater_equal<std::common_type_t<Scalar,typename View::value_type>>, Scalar,
                        ScalarOnRight>;

template <typename Scalar, zipper::concepts::QualifiedViewDerived View,
          bool ScalarOnRight>
using ScalarLessEqualView =
    ScalarOperationView<View, std::less_equal<std::common_type_t<Scalar,typename View::value_type>>, Scalar, ScalarOnRight>;

// logical and preserves false
template <typename Scalar, zipper::concepts::QualifiedViewDerived View,
          bool ScalarOnRight>
using ScalarLogicalAndView = ScalarOperationView<View, std::logical_and<std::common_type_t<Scalar,typename View::value_type>>,
                                                 Scalar, ScalarOnRight, true>;

template <typename Scalar, zipper::concepts::QualifiedViewDerived View,
          bool ScalarOnRight>
using ScalarLogicalOrView =
    ScalarOperationView<View, std::logical_or<std::common_type_t<Scalar,typename View::value_type>>, Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::QualifiedViewDerived View,
          bool ScalarOnRight>
using ScalarBitAndView =
    ScalarOperationView<View, std::bit_and<std::common_type_t<Scalar,typename View::value_type>>, Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::QualifiedViewDerived View,
          bool ScalarOnRight>
using ScalarBitOrView =
    ScalarOperationView<View, std::bit_or<std::common_type_t<Scalar,typename View::value_type>>, Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::QualifiedViewDerived View,
          bool ScalarOnRight>
using ScalarBitXorView =
    ScalarOperationView<View, std::bit_xor<std::common_type_t<Scalar,typename View::value_type>>, Scalar, ScalarOnRight>;

}  // namespace zipper::views::unary

#endif
