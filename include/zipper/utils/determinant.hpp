
#if !defined(ZIPPER_UTILS_DETERMINANT_HPP)
#define ZIPPER_UTILS_DETERMINANT_HPP
#include <zipper/Matrix.hpp>
#include <zipper/concepts/MatrixBaseDerived.hpp>

#include "detail/tuple_to_array.hpp"
#include "zipper/detail/extents/is_cubic.hpp"
#include "zipper/detail/extents/swizzle_extents.hpp"
#include "zipper/views/reductions/Determinant.hpp"
namespace zipper::utils {
template <zipper::concepts::MatrixBaseDerived D>
auto determinant(const D& d) {
    zipper::detail::extents::throw_if_not_cubic(d.extents());
    return zipper::views::reductions::Determinant(d.view())();
}
}  // namespace zipper::utils
#endif
