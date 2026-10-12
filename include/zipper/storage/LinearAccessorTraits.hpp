#if !defined(ZIPPER_STORAGE_LINEARACCESORTRAITS_HPP)
#define ZIPPER_STORAGE_LINEARACCESORTRAITS_HPP
#include "zipper/detail/Features.hpp"

namespace zipper::storage {

using AccessFeatures = zipper::detail::AccessFeatures;
using ShapeFeatures = zipper::detail::ShapeFeatures;

template <typename T> struct LinearAccessorTraits;

template <AccessFeatures AF, ShapeFeatures SF, bool IsContiguous = true>
struct BasicLinearAccessorTraits {
  constexpr static AccessFeatures access_features = AF;
  constexpr static ShapeFeatures shape_features = SF;
  /// Elements live in a raw contiguous buffer exposed by data(), so the
  /// accessor's k-th element is data()[k].
  constexpr static bool is_contiguous = IsContiguous;
};

} // namespace zipper::storage
#endif
