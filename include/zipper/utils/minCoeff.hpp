#if !defined(ZIPPER_UTILS_MINCOEFF_HPP)
#define ZIPPER_UTILS_MINCOEFF_HPP
#include <zipper/ZipperBase.hpp>
#include <zipper/concepts/ZipperBaseDerived.hpp>

#include "detail/tuple_to_array.hpp"
namespace zipper::utils {

namespace detail {
template <zipper::concepts::ZipperBaseDerived D, bool ReturnMultiIndex>

auto _minCoeff(const D& d) {
    constexpr static index_type rank = D::extents_traits::rank;
    using T = typename D::value_type;

    T min = std::numeric_limits<T>::max();
    // hopefully the compiler will alide this
    using AT = std::array<index_type, rank>;
    AT arr;
    for (const auto& i :
         zipper::detail::extents::all_extents_indices(d.extents())) {
        T v = d(i);
        if (v < min) {
            min = v;
            if constexpr (ReturnMultiIndex) {
                arr = detail::tuple_to_array(i);
            }
        }
    }
    if constexpr (ReturnMultiIndex) {
        return std::make_pair(min, arr);
    } else {
        return min;
    }
}
}  // namespace detail

template <zipper::concepts::ZipperBaseDerived D>
auto minCoeff(const D& d) {
    return detail::_minCoeff<D, false>(d);
}
template <zipper::concepts::ZipperBaseDerived D>
auto minCoeffWithIndex(const D& d) {
    return detail::_minCoeff<D, true>(d);
}
}  // namespace zipper::utils
#endif
