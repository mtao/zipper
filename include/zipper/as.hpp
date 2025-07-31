

#if !defined(ZIPPER_AS_HPP)
#define ZIPPER_AS_HPP
#include "concepts/ZipperBaseDerived.hpp"
#include "views/unary/IdentityView.hpp"

namespace zipper {
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_array(ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    constexpr static bool is_const = ZipperDerived::view_type::traits::is_writable || std::is_const_v<ZipperDerived>;
    using ViewC = std::conditional_t< is_const, const View, View>;
    // make it non-owning
    using V = views::unary::IdentityView<ViewC>;
    return ArrayBase(V(v.view()));
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_tensor(ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    constexpr static bool is_const = ZipperDerived::view_type::traits::is_writable || std::is_const_v<ZipperDerived>;
    using ViewC = std::conditional_t< is_const, const View, View>;
    // make it non-owning
    using V = views::unary::IdentityView<ViewC>;
    return TensorBase(V(v.view()));
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_matrix(ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    constexpr static bool is_const = ZipperDerived::view_type::traits::is_writable || std::is_const_v<ZipperDerived>;
    using ViewC = std::conditional_t< is_const, const View, View>;
    // make it non-owning
    using V = views::unary::IdentityView<ViewC>;
    return MatrixBase(V(v.view()));
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_vector(ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    constexpr static bool is_const = ZipperDerived::view_type::traits::is_writable || std::is_const_v<ZipperDerived>;
    using ViewC = std::conditional_t< is_const, const View, View>;
    // make it non-owning
    using V = views::unary::IdentityView<ViewC>;
    return VectorBase(V(v.view()));
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_form(ZipperDerived& v) {
    using View = ZipperDerived::view_type;

    constexpr static bool is_const = ZipperDerived::view_type::traits::is_writable || std::is_const_v<ZipperDerived>;
    using ViewC = std::conditional_t< is_const, const View, View>;
    // make it non-owning
    using V = views::unary::IdentityView<ViewC>;
    return FormBase(V(v.view()));
}

}  // namespace zipper
#endif
