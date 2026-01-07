

#if !defined(ZIPPER_AS_HPP)
#define ZIPPER_AS_HPP
#include "concepts/ZipperBaseDerived.hpp"

namespace zipper {

// As functions for each basic type

template <concepts::ZipperBaseDerived ZipperDerived>
auto as_array(ZipperDerived &v) {
  using View = ZipperDerived::view_type;
  constexpr static bool is_const =
      ZipperDerived::view_type::traits::is_writable ||
      std::is_const_v<ZipperDerived>;
  using ViewC = std::conditional_t<is_const, const View, View>;
  return ArrayBase<ViewC &>(v.view());
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_tensor(ZipperDerived &v) {
  using View = ZipperDerived::view_type;
  constexpr static bool is_const =
      ZipperDerived::view_type::traits::is_writable ||
      std::is_const_v<ZipperDerived>;
  using ViewC = std::conditional_t<is_const, const View, View>;
  return TensorBase<ViewC &>(v.view());
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_matrix(ZipperDerived &v) {
  using View = ZipperDerived::view_type;
  constexpr static bool is_const =
      ZipperDerived::view_type::traits::is_writable ||
      std::is_const_v<ZipperDerived>;
  using ViewC = std::conditional_t<is_const, const View, View>;
  return MatrixBase<ViewC &>(v.view());
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_vector(ZipperDerived &v) {
  using View = ZipperDerived::view_type;
  constexpr static bool is_const =
      ZipperDerived::view_type::traits::is_writable ||
      std::is_const_v<ZipperDerived>;
  using ViewC = std::conditional_t<is_const, const View, View>;
  return VectorBase<ViewC &>(v.view());
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_form(ZipperDerived &v) {
  using View = ZipperDerived::view_type;
  constexpr static bool is_const =
      ZipperDerived::view_type::traits::is_writable ||
      std::is_const_v<ZipperDerived>;
  using ViewC = std::conditional_t<is_const, const View, View>;
  return FormBase<ViewC &>(v.view());
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_array(const ZipperDerived &v) {
  using View = ZipperDerived::view_type;
  return ArrayBase<const View>(v.view());
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_tensor(const ZipperDerived &v) {
  using View = ZipperDerived::view_type;
  return TensorBase<const View>(v.view());
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_matrix(const ZipperDerived &v) {
  using View = ZipperDerived::view_type;
  return MatrixBase<const View>(v.view());
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_vector(const ZipperDerived &v) {
  using View = ZipperDerived::view_type;
  return VectorBase<const View>(v.view());
}
template <concepts::ZipperBaseDerived ZipperDerived>
auto as_form(const ZipperDerived &v) {
  using View = ZipperDerived::view_type;

  return FormBase<const View>(v.view());
}

} // namespace zipper
#endif
