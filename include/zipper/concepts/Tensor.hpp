#if !defined(ZIPPER_CONCEPTS_TENSOR_HPP)
#define ZIPPER_CONCEPTS_TENSOR_HPP
#include <type_traits>

#include "Expression.hpp"
#include "Extents.hpp"

namespace zipper {
template <concepts::Expression T> class TensorBase;
template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
class Tensor_;
template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
class TensorSpan_;
} // namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename> struct IsTensor : std::false_type {};
template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
struct IsTensor<Tensor_<ValueType, Extents, LeftMajor>> : std::true_type {};
template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
struct IsTensor<TensorSpan_<ValueType, Extents, LeftMajor>> : std::true_type {};

template <typename> struct IsTensorBase : std::false_type {};
template <typename T> struct IsTensorBase<TensorBase<T>> : std::true_type {};
} // namespace detail
//
template <typename T>
concept Tensor = detail::IsTensor<T>::value || detail::IsTensorBase<T>::value;

/// Anything derived from expression can be treated like a tensor
template <typename T>
concept TensorExpression = concepts::Expression<T>;

} // namespace zipper::concepts
#endif
