#if !defined(ZIPPER_DETAIL_SLICE_HELPERS_HPP)
#define ZIPPER_DETAIL_SLICE_HELPERS_HPP

#include "zipper/detail/constexpr_arithmetic.hpp"
#include "zipper/types.hpp"

namespace zipper::detail {

inline index_type first_of(const index_type &index) { return index; }
inline index_type last_of(const index_type &index) { return index; }
template <typename OffsetType, typename ExtentType, typename StrideType>
inline index_type
first_of(const strided_slice<OffsetType, ExtentType, StrideType> &s) {
  return s.offset;
}
template <typename OffsetType, typename ExtentType, typename StrideType>
inline index_type
last_of(const strided_slice<OffsetType, ExtentType, StrideType> &s) {
  ConstexprArithmetic o(s.offset);
  ConstexprArithmetic e(s.extent);
  return e - o;
}
template <typename OffsetType, typename ExtentType, typename StrideType>
inline index_type
stride_of(const strided_slice<OffsetType, ExtentType, StrideType> &s) {
  return s.stride;
}

} // namespace zipper::detail

#endif
