#if !defined(ZIPPER_VIEWS_BINARYARITHMETICVIEW_HPP)
#define ZIPPER_VIEWS_BINARYARITHMETICVIEW_HPP

#include <functional>

#include "OperationView.hpp"
#include "detial/minmax.hpp"

namespace zipper::views::binary {
namespace detail {
template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
struct scalar_type {
    using ATraits = views::detail::ViewTraits<ViewA>;
    using BTraits = views::detail::ViewTraits<ViewB>;

    using a_value_type = ATraits::value_type;
    using b_value_type = BTraits::value_type;
    static_assert(std::is_same_v<a_value_type, b_value_type>);
    using type = a_value_type;
};

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using scalar_type_t = scalar_type<ViewA, ViewB>::type;
}  // namespace detail

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using PlusView =
    OperationView<ViewA, ViewB, std::plus<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using MinusView =
    OperationView<ViewA, ViewB,
                  std::minus<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using MultipliesView =
    OperationView<ViewA, ViewB,
                  std::multiplies<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using DividesView =
    OperationView<ViewA, ViewB,
                  std::divides<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using ModulusView =
    OperationView<ViewA, ViewB,
                  std::modulus<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using EqualsToView =
    OperationView<ViewA, ViewB,
                  std::equal_to<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using NotEqualsToView =
    OperationView<ViewA, ViewB,
                  std::not_equal_to<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using GreaterView =
    OperationView<ViewA, ViewB,
                  std::greater<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using LessView =
    OperationView<ViewA, ViewB, std::less<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using GreaterEqualView =
    OperationView<ViewA, ViewB,
                  std::greater_equal<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using LessEqualView =
    OperationView<ViewA, ViewB,
                  std::less_equal<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using LogicalAndView =
    OperationView<ViewA, ViewB,
                  std::logical_and<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using LogicalOrView =
    OperationView<ViewA, ViewB,
                  std::logical_or<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using BitAndView =
    OperationView<ViewA, ViewB,
                  std::bit_and<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using BitOrView =
    OperationView<ViewA, ViewB,
                  std::bit_or<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using BitXorView =
    OperationView<ViewA, ViewB,
                  std::bit_xor<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using MinView = OperationView<ViewA, ViewB,
                              detail::min<detail::scalar_type_t<ViewA, ViewB>>>;

template <concepts::ViewDerived ViewA, concepts::ViewDerived ViewB>
using MaxView = OperationView<ViewA, ViewB,
                              detail::max<detail::scalar_type_t<ViewA, ViewB>>>;
}  // namespace zipper::views::binary

#endif
