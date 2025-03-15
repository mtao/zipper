#if !defined(UVL_VIEWS_UNARY_SCALARARITHMETICVIEW_HPP)
#define UVL_VIEWS_UNARY_SCALARARITHMETICVIEW_HPP

#include <functional>
#include "ScalarOperationView.hpp"

namespace uvl::views::unary {
template <typename A, concepts::ViewDerived B, bool ScalarOnRight>
using ScalarPlusView =
    ScalarOperationView<B, std::plus<A>, A, ScalarOnRight>;

template <typename A, concepts::ViewDerived B, bool ScalarOnRight>
using ScalarMinusView =
    ScalarOperationView<B, std::minus<A>, A, ScalarOnRight>;

template <typename A, concepts::ViewDerived B, bool ScalarOnRight>
using ScalarMultipliesView =
    ScalarOperationView<B, std::multiplies<A>, A, ScalarOnRight>;

template <typename A, concepts::ViewDerived B, bool ScalarOnRight>
using ScalarDividesView =
    ScalarOperationView<B, std::divides<A>, A, ScalarOnRight>;

template <typename A, concepts::ViewDerived B, bool ScalarOnRight>using ScalarModulusView =    ScalarOperationView<B, 
    std::modulus<A>, A, ScalarOnRight>;


template <typename A, concepts::ViewDerived B, bool ScalarOnRight>using ScalarEqualsToView =    ScalarOperationView<B,std::equal_to<A>, A, ScalarOnRight>;
template <typename A, concepts::ViewDerived B, bool ScalarOnRight>using ScalarNotEqualsToView =    ScalarOperationView<B,std::not_equal_to<A>, A, ScalarOnRight>;
template <typename A, concepts::ViewDerived B, bool ScalarOnRight>using ScalarGreaterView =    ScalarOperationView<B,std::greater<A>, A, ScalarOnRight>;
template <typename A, concepts::ViewDerived B, bool ScalarOnRight>using ScalarLessView =    ScalarOperationView<B,std::less<A>, A, ScalarOnRight>;
template <typename A, concepts::ViewDerived B, bool ScalarOnRight>using ScalarGreaterEqualView =    ScalarOperationView<B,std::greater_equal<A>, A, ScalarOnRight>;
template <typename A, concepts::ViewDerived B, bool ScalarOnRight>using ScalarLessEqualView =    ScalarOperationView<B,std::less_equal<A>, A, ScalarOnRight>;
template <typename A, concepts::ViewDerived B, bool ScalarOnRight>using ScalarLogicalAndView =    ScalarOperationView<B,std::logical_and<A>, A, ScalarOnRight>;
template <typename A, concepts::ViewDerived B, bool ScalarOnRight>using ScalarLogicalOrView =    ScalarOperationView<B,std::logical_or<A>, A, ScalarOnRight>;
template <typename A, concepts::ViewDerived B, bool ScalarOnRight>using ScalarLogicalNotView =    ScalarOperationView<B,std::logical_not<A>, A, ScalarOnRight>;
template <typename A, concepts::ViewDerived B, bool ScalarOnRight>using ScalarBitAndView =    ScalarOperationView<B,std::bit_and<A>, A, ScalarOnRight>;
template <typename A, concepts::ViewDerived B, bool ScalarOnRight>using ScalarBitOrView =    ScalarOperationView<B,std::bit_or<A>, A, ScalarOnRight>;
template <typename A, concepts::ViewDerived B, bool ScalarOnRight>using ScalarBitXorView =    ScalarOperationView<B,std::bit_xor<A>, A, ScalarOnRight>;
template <typename A, concepts::ViewDerived B, bool ScalarOnRight>using ScalarBitNotView =    ScalarOperationView<B,std::bit_not<A>, A, ScalarOnRight>;
  

}  // namespace uvl::views

#endif
