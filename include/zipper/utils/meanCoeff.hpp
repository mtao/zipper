#if !defined(ZIPPER_UTILS_MEANCOEFF_HPP)
#define ZIPPER_UTILS_MEANCOEFF_HPP
#include <zipper/ZipperBase.hpp>
#include <zipper/concepts/ZipperBaseDerived.hpp>
#include <zipper/views/reductions/CoefficientSum.hpp>

#include "detail/tuple_to_array.hpp"
namespace zipper::utils {

template <zipper::concepts::ZipperBaseDerived D>
auto meanCoeff(const D& d) {
    zipper::views::reductions::CoefficientSum s(d.view());
    return s() / typename D::value_type(d.view().size());
}
}  // namespace zipper::utils
#endif
