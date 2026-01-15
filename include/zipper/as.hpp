#if !defined(ZIPPER_AS_HPP)
#define ZIPPER_AS_HPP
#include "concepts/ZipperBaseDerived.hpp"

namespace zipper {

// As functions for each basic type

#define AS_IMPL(NAME_LOWER, NAME_UPPER)                                        \
  template <concepts::ZipperBaseDerived ZipperDerived>                         \
  auto as_##NAME_LOWER(ZipperDerived &v) {                                     \
    using View = ZipperDerived::view_type;                                     \
    constexpr static bool is_const =                                           \
        ZipperDerived::view_type::traits::is_writable ||                       \
        std::is_const_v<ZipperDerived>;                                        \
    using ViewC = std::conditional_t<is_const, const View, View>;              \
    return NAME_UPPER##Base<ViewC &>(v.view());                                \
  }                                                                            \
  template <concepts::ZipperBaseDerived ZipperDerived>                         \
  auto as_##NAME_LOWER(const ZipperDerived &v) {                               \
    using View = ZipperDerived::view_type;                                     \
    return NAME_UPPER##Base<const View>(v.view());                             \
  }

AS_IMPL(array, Array)
AS_IMPL(vector, Vector)
AS_IMPL(matrix, Matrix)
AS_IMPL(form, Form)
AS_IMPL(tensor, Tensor)

} // namespace zipper
#endif
