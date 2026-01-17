#if !defined(ZIPPER_CONCEPTS_ZIPPERBASE_DERIVED_HPP)
#define ZIPPER_CONCEPTS_ZIPPERBASE_DERIVED_HPP
#include <type_traits>

#include "Expression.hpp"

namespace zipper {
template <template <concepts::QualifiedExpression> typename DerivedT,
          concepts::QualifiedExpression Expression>
class ZipperBase;
} // namespace zipper
namespace zipper::concepts {
namespace detail {

template <typename> struct ZipperBaseDerived : public std::false_type {};
template <template <concepts::QualifiedExpression> typename DerivedT,
          concepts::QualifiedExpression Expression>
struct ZipperBaseDerived<ZipperBase<DerivedT, Expression>> : std::true_type {};

} // namespace detail

template <typename T>
concept ZipperBaseDerived = detail::ZipperBaseDerived<std::decay_t<T>>::value;
} // namespace zipper::concepts
#endif
