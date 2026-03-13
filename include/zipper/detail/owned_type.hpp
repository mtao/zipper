#if !defined(ZIPPER_DETAIL_OWNED_TYPE_HPP)
#define ZIPPER_DETAIL_OWNED_TYPE_HPP

#include <type_traits>

namespace zipper::detail {

/// Type trait that computes the "owned" version of an expression type.
///
/// Every expression class defines a `make_owned()` method that recursively
/// deep-copies child expressions so the result owns all its data (no
/// dangling references).  `owned_t<T>` is the return type of that method.
///
/// For qualified types (const T, T&, const T&), the cv-ref qualifiers are
/// stripped before dispatching to the expression's make_owned().
template <typename T>
using owned_t = decltype(std::declval<const std::decay_t<T>&>().make_owned());

} // namespace zipper::detail

#endif
