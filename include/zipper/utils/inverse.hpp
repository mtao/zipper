#if !defined(ZIPPER_UTILS_INVERSE_HPP)
#define ZIPPER_UTILS_INVERSE_HPP
#include <zipper/Matrix.hpp>
#include <zipper/concepts/Matrix.hpp>

#include "detail/tuple_to_array.hpp"
#include "zipper/utils/extents/is_compatible.hpp"
#include "zipper/detail/extents/swizzle_extents.hpp"
#include "zipper/expression/reductions/Determinant.hpp"
namespace zipper::utils {

namespace detail {
template <zipper::concepts::Matrix D>
auto inverse2d(const D& M) {
    zipper::utils::extents::throw_if_not_compatible<2, 2>(M.extents());
    auto extents = zipper::detail::extents::swizzle_extents<1, 0>(M.extents());
    using extents_type = std::decay_t<decltype(extents)>;

    using T = typename D::value_type;

    constexpr static index_type static_rows = extents_type::static_extent(0);
    constexpr static index_type static_cols = extents_type::static_extent(1);
    Matrix<T, static_rows, static_cols> I(extents);
    // a b
    // c d
    // 1/det ( d -b )
    //       ( -c a )
    const T& a = M(0, 0);
    const T& b = M(0, 1);
    const T& c = M(1, 0);
    const T& d = M(1, 1);
    T det = zipper::expression::reductions::Determinant(M.expression())();
    I(0, 0) = d / det;
    I(0, 1) = -b / det;
    I(1, 0) = -c / det;
    I(1, 1) = a / det;
    return I;
}
template <zipper::concepts::Matrix D>
auto inverse3d(const D& M) {
    zipper::utils::extents::throw_if_not_compatible<3, 3>(M.extents());
    auto extents = zipper::detail::extents::swizzle_extents<1, 0>(M.extents());
    using extents_type = std::decay_t<decltype(extents)>;

    using T = typename D::value_type;

    constexpr static index_type static_rows = extents_type::static_extent(0);
    constexpr static index_type static_cols = extents_type::static_extent(1);
    Matrix<T, static_rows, static_cols> I(extents);
    // a b
    // c d
    // 1/det ( d -b )
    //       ( -c a )
    T det = zipper::expression::reductions::Determinant(M.expression())();
    auto cofactor = [](const auto& v_, index_type r, index_type c) {
        auto det2 = [](const auto& v, index_type r0, index_type c0,
                       index_type r1, index_type c1) {
            const auto val = zipper::expression::reductions::detail::det2(
                v(r0, c0), v(r0, c1), v(r1, c0), v(r1, c1));
            return val;
        };
        return det2(v_, (r + 1) % 3, (c + 1) % 3, (r + 2) % 3, (c + 2) % 3);
    };
    for (index_type r = 0; r < 3; ++r) {
        for (index_type c = 0; c < 3; ++c) {
            I(r, c) = cofactor(M, c, r) / det;
        }
    }
    return I;
}
}  // namespace detail

template <zipper::concepts::Matrix D>
auto inverse(const D& d) {
    const auto& extents = d.extents();
    if (zipper::utils::extents::is_compatible<2, 2>(extents)) {
        return detail::inverse2d(d);
    } else if (zipper::utils::extents::is_compatible<3, 3>(extents)) {
        return detail::inverse3d(d);
    }
    // TODO: implement a general inverse view perhaps?
    throw std::runtime_error("Not implemented");
}
}  // namespace zipper::utils
#endif
