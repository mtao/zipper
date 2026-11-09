#if !defined(ZIPPER_AS_HPP)
#define ZIPPER_AS_HPP
// The as_* conversions (as_array, as_vector, ...) are defined by their target
// base's header (as_array by ArrayBase.hpp, ...); include that header to use
// one.  Their shared machinery is in detail/as_conversion.hpp.
#include "detail/as_conversion.hpp"
#endif
