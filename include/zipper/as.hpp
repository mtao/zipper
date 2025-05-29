

#if !defined(ZIPPER_AS_HPP)
#define ZIPPER_AS_HPP
#include "concepts/ZipperBaseDerived.hpp"
#include "views/unary/IdentityView.hpp"

namespace zipper {
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_array(const ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    // make it non-owning
    using V = views::unary::IdentityView<View>;
    return ArrayBase(V(v.view()));
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_tensor(const ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    // make it non-owning
    using V = views::unary::IdentityView<View>;
    return TensorBase(V(v.view()));
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_matrix(const ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    // make it non-owning
    using V = views::unary::IdentityView<View>;
    return MatrixBase(V(v.view()));
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_vector(const ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    // make it non-owning
    using V = views::unary::IdentityView<View>;
    return VectorBase(V(v.view()));
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_form(const ZipperDerived& v) {
    using View = ZipperDerived::view_type;
    // make it non-owning
    using V = views::unary::IdentityView<View>;
    return FormBase(V(v.view()));
}

}  // namespace zipper
#endif
