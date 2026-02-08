#if !defined(ZIPPER_CONCEPTS_ZIPPER_HPP)
#define ZIPPER_CONCEPTS_ZIPPER_HPP

#include "detail/IsZipperBase.hpp"

// Each concept header (Array.hpp, Form.hpp, etc.) provides its own
// IsZipperBase specialization, so including them here pulls in all
// built-in registrations.  External code can add new ZipperBase
// families by specializing IsZipperBase without modifying this file.
#include "Array.hpp"
#include "Form.hpp"
#include "Matrix.hpp"
#include "Tensor.hpp"
#include "Vector.hpp"

namespace zipper::concepts {

template <typename T>
concept Zipper = detail::IsZipperBase<std::decay_t<T>>::value;

} // namespace zipper::concepts
#endif
