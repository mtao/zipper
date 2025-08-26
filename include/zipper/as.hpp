

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
    //using V = views::unary::IdentityView<ViewC>;
    //return ArrayBase(V(v.view()));
    return ArrayBase<ViewC&>(v.view());
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_tensor(ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    constexpr static bool is_const = ZipperDerived::view_type::traits::is_writable || std::is_const_v<ZipperDerived>;
    using ViewC = std::conditional_t< is_const, const View, View>;
    // make it non-owning
    //using V = views::unary::IdentityView<ViewC>;
    return TensorBase<ViewC&>(v.view());
    //return TensorBase(V(v.view()));
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_matrix(ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    constexpr static bool is_const = ZipperDerived::view_type::traits::is_writable || std::is_const_v<ZipperDerived>;
    using ViewC = std::conditional_t< is_const, const View, View>;
    // make it non-owning
    //using V = views::unary::IdentityView<ViewC>;
    return MatrixBase<ViewC&>(v.view());
    //return MatrixBase(V(v.view()));
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_vector(ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    constexpr static bool is_const = ZipperDerived::view_type::traits::is_writable || std::is_const_v<ZipperDerived>;
    using ViewC = std::conditional_t< is_const, const View, View>;
    // make it non-owning
    //using V = views::unary::IdentityView<ViewC>;
    return VectorBase<ViewC&>(v.view());
    //return VectorBase(V(v.view()));
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_form(ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    constexpr static bool is_const = ZipperDerived::view_type::traits::is_writable || std::is_const_v<ZipperDerived>;
    using ViewC = std::conditional_t< is_const, const View, View>;
    // make it non-owning
    //using V = views::unary::IdentityView<ViewC>;
    return FormBase<ViewC&>(v.view());
    //return FormBase(V(v.view()));
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_array(const ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    // make it non-owning
    //using V = views::unary::IdentityView<const View>;
    return ArrayBase<const View>(v.view());
    //return ArrayBase(V(v.view()));
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_tensor(const ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    // make it non-owning
    //using V = views::unary::IdentityView<const View>;
    return TensorBase<const View>(v.view());
    //return TensorBase(V(v.view()));
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_matrix(const ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    // make it non-owning
    //using V = views::unary::IdentityView<const View>;
    return MatrixBase<const View>(v.view());
    //return MatrixBase(V(v.view()));
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_vector(const ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    // make it non-owning
    //using V = views::unary::IdentityView<const View>;
    return VectorBase<const View>(v.view());
    //return VectorBase(V(v.view()));
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_form(const ZipperDerived& v) {
    using View = ZipperDerived::view_type;

    // make it non-owning
    //using V = views::unary::IdentityView<const View>;
    return FormBase<const View>(v.view());
    //return FormBase(V(v.view()));
}

}  // namespace zipper
#endif
