#if !defined(ZIPPER_STORAGE_DENSEDATA_HPP)
#define ZIPPER_STORAGE_DENSEDATA_HPP
#include "DynamicDenseData.hpp"
#include "StaticDenseData.hpp"
#include "zipper/detail//ExtentsTraits.hpp"

template <typename ValueType, typename Extents>
using DenseData = std::conditional_t<
    zipper::detail::ExtentsTraits<Extents>::is_static,
    StaticDenseData<ValueType, zipper::detail::template ExtentsTraits<
                                   Extents>::static_size>,
    DynamicDenseData<ValueType>>;
}  // namespace zipper::storage
#endif

