#if !defined(ZIPPER_ARRAY_HPP)
#define ZIPPER_ARRAY_HPP

#include "ArrayBase.hpp"
#include "concepts/ArrayBaseDerived.hpp"
#include "concepts/ExtentsType.hpp"
#include "concepts/MatrixViewDerived.hpp"
#include "concepts/VectorViewDerived.hpp"
#include "storage/PlainObjectStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

namespace detail {
template <typename ValueType, concepts::ExtentsType Extents,
          bool LeftMajor = true>
class Array_ : public ArrayBase<storage::PlainObjectStorage<
                   ValueType, Extents, storage::tensor_layout<LeftMajor>>> {
public:
  using layout_type = storage::tensor_layout<LeftMajor>;
  using Base =
      ArrayBase<storage::PlainObjectStorage<ValueType, Extents, layout_type>>;
  using Base::view;
  using view_type = Base::view_type;
  using value_type = Base::value_type;
  using extents_type = Base::extents_type;
  using Base::extent;
  using Base::extents;
  using span_type =
      ArrayBase<storage::SpanStorage<ValueType, Extents, layout_type>>;

  Array_(const Array_ &o) = default;
  Array_(Array_ &&o) = default;
  auto operator=(const Array_ &o) -> Array_ & = default;
  auto operator=(Array_ &&o) -> Array_ & = default;
  template <concepts::ViewDerived Other>
  Array_(const Other &other) : Base(other) {}
  template <concepts::ArrayBaseDerived Other>
  Array_(const Other &other) : Base(other) {}
  template <typename... Args>
  Array_(Args &&...args)
    requires((std::is_convertible_v<Args, index_type> && ...))
      : Base(Extents(std::forward<Args>(args)...)) {}
  template <index_type... indices>
  Array_(const zipper::extents<indices...> &e) : Base(e) {}
};

template <concepts::MatrixViewDerived MB>
Array_(const MB &o)
    -> Array_<typename MB::value_type, typename MB::extents_type>;

template <concepts::VectorViewDerived MB>
Array_(const MB &o)
    -> Array_<typename MB::value_type, typename MB::extents_type>;

} // namespace detail
template <typename ValueType, index_type... Indxs>
using Array = detail::Array_<ValueType, zipper::extents<Indxs...>>;

namespace concepts::detail {

template <typename ValueType, concepts::ExtentsType Extents, bool LeftMajor>
struct ArrayBaseDerived<zipper::detail::Array_<ValueType, Extents, LeftMajor>>
    : std::true_type {};
} // namespace concepts::detail

} // namespace zipper
#endif
