#if !defined(ZIPPER_UTILS_MAXCOEFF_HPP)
#define ZIPPER_UTILS_MAXCOEFF_HPP
#include <zipper/ZipperBase.hpp>
#include <zipper/concepts/Zipper.hpp>

#include "detail/tuple_to_array.hpp"
namespace zipper::utils {

namespace detail {
template <zipper::concepts::Zipper D, bool ReturnMultiIndex>

auto _maxCoeff(const D& d) {
    constexpr static index_type rank = D::extents_traits::rank;
    using T = typename D::value_type;

    T max = std::numeric_limits<T>::lowest();
    // hopefully the compiler will alide this
    using AT = std::array<index_type, rank>;
    AT arr;
    for (const auto& i :
         zipper::utils::extents::all_extents_indices(d.extents())) {
        T v = std::apply(d, i);
        if (v > max) {
            max = v;
            if constexpr (ReturnMultiIndex) {
                arr = detail::tuple_to_array(i);
            }
        }
    }
    if constexpr (ReturnMultiIndex) {
        return std::make_pair(max, arr);
    } else {
        return max;
    }
}
}  // namespace detail
template <zipper::concepts::Zipper D>
auto maxCoeff(const D& d) {
    return detail::_maxCoeff<D, false>(d);
}
template <zipper::concepts::Zipper D>
auto maxCoeffWithIndex(const D& d) {
    return detail::_maxCoeff<D, true>(d);
}
}  // namespace zipper::utils
#endif
