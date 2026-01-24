#if !defined(ZIPPER_STORAGE_LINEARACCESORTRAITS_HPP)
#define ZIPPER_STORAGE_LINEARACCESORTRAITS_HPP
#include "zipper/detail/Features.hpp"

namespace zipper::storage {

using AccessFeatures = zipper::detail::AccessFeatures;
using ShapeFeatures = zipper::detail::ShapeFeatures;

template <typename T> struct LinearAccessorTraits;

template <AccessFeatures AF, ShapeFeatures SF>
struct BasicLinearAccessorTraits {
  constexpr static AccessFeatures access_features = AF;
  constexpr static ShapeFeatures shape_features = SF;
};

} // namespace zipper::storage
#endif
