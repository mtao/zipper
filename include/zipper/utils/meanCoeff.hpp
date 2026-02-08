#if !defined(ZIPPER_UTILS_MEANCOEFF_HPP)
#define ZIPPER_UTILS_MEANCOEFF_HPP
#include <zipper/ZipperBase.hpp>
#include <zipper/concepts/Zipper.hpp>
#include <zipper/expression/reductions/CoefficientSum.hpp>

#include "detail/tuple_to_array.hpp"
namespace zipper::utils {

template <zipper::concepts::Zipper D>
auto meanCoeff(const D& d) {
    zipper::expression::reductions::CoefficientSum s(d.expression());
    return s() / typename D::value_type(d.expression().size());
}
}  // namespace zipper::utils
#endif
