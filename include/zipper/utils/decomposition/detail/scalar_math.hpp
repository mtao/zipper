#if !defined(ZIPPER_UTILS_DECOMPOSITION_DETAIL_SCALAR_MATH_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_DETAIL_SCALAR_MATH_HPP

#include <zipper/utils/scalar_math.hpp>

namespace zipper::utils::decomposition::detail {

template <typename T>
concept orthogonal_decomposition_scalar =
    utils::scalar_math::orthogonal_decomposition_scalar<T>;

namespace scalar_math = utils::scalar_math;

} // namespace zipper::utils::decomposition::detail

#endif
