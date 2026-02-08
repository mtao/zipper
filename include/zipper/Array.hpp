#if !defined(ZIPPER_ARRAY_HPP)
#define ZIPPER_ARRAY_HPP

#include "ArrayBase.hpp"
#include "concepts/Array.hpp"
#include "concepts/Extents.hpp"
#include "concepts/Matrix.hpp"
#include "concepts/Vector.hpp"
#include "zipper/expression/nullary/MDArray.hpp"
#include "zipper/types.hpp"
namespace zipper {

namespace detail {
template <typename ValueType, concepts::Extents Extents,
          bool LeftMajor = true>
class Array_ : public ArrayBase<expression::nullary::MDArray<
                   ValueType, Extents, storage::tensor_layout<LeftMajor>,
                   default_accessor_policy<ValueType>>> {
public:
  using layout_type = storage::tensor_layout<LeftMajor>;
  using expression_type =
      expression::nullary::MDArray<ValueType, Extents, layout_type,
                                   default_accessor_policy<ValueType>>;
  using Base = ArrayBase<expression_type>;
  using Base::expression;
  using value_type = Base::value_type;
  using extents_type = Base::extents_type;
  using Base::extent;
  using Base::extents;
  using span_type =
      ArrayBase<expression::nullary::MDSpan<ValueType, Extents, layout_type,
                                            default_accessor_policy<ValueType>>>;

  Array_(const Array_ &o) = default;
  Array_(Array_ &&o) = default;
  auto operator=(const Array_ &o) -> Array_ & = default;
  auto operator=(Array_ &&o) -> Array_ & = default;
  template <concepts::Expression Other>
  Array_(const Other &other) : Base(other) {}
  template <concepts::Array Other>
  Array_(const Other &other) : Base(other) {}
  template <typename... Args>
  Array_(Args &&...args)
    requires((std::is_convertible_v<Args, index_type> && ...))
      : Base(Extents(std::forward<Args>(args)...)) {}
  template <index_type... indices>
  Array_(const zipper::extents<indices...> &e) : Base(e) {}
};

template <concepts::Matrix MB>
Array_(const MB &o)
    -> Array_<typename MB::value_type, typename MB::extents_type>;

template <concepts::Vector MB>
Array_(const MB &o)
    -> Array_<typename MB::value_type, typename MB::extents_type>;

} // namespace detail
template <typename ValueType, index_type... Indxs>
using Array = detail::Array_<ValueType, zipper::extents<Indxs...>>;

namespace concepts::detail {

template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
struct IsArray<zipper::detail::Array_<ValueType, Extents, LeftMajor>>
    : std::true_type {};
template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
struct IsZipperBase<zipper::detail::Array_<ValueType, Extents, LeftMajor>>
    : std::true_type {};
} // namespace concepts::detail

} // namespace zipper
#endif
