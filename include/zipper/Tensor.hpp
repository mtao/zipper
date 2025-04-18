#if !defined(ZIPPER_TENSOR_HPP)
#define ZIPPER_TENSOR_HPP

#include "TensorBase.hpp"
#include "TensorSpan.hpp"
#include "concepts/ExtentsType.hpp"
#include "concepts/TensorBaseDerived.hpp"
#include "storage/PlainObjectStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, concepts::ExtentsType Extents, bool LeftMajor>
class Tensor_
    : public TensorBase<storage::PlainObjectStorage<
          ValueType, Extents,
          std::conditional_t<LeftMajor, std::experimental::layout_left,
                             std::experimental::layout_right>>> {
   public:
    using layout_type =
        std::conditional_t<LeftMajor, std::experimental::layout_left,
                           std::experimental::layout_right>;
    using Base = TensorBase<
        storage::PlainObjectStorage<ValueType, Extents, layout_type>>;
    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using Base::extent;
    using Base::extents;
    using span_type = TensorSpan_<ValueType, Extents>;

    template <concepts::ViewDerived Other>
    Tensor_(const Other& other) : Base(other) {}
    template <concepts::TensorBaseDerived Other>
    Tensor_(const Other& other) : Base(other) {}
    template <typename... Args>
    Tensor_(Args&&... args)
        requires((std::is_convertible_v<Args, index_type> && ...))
        : Base(Extents(std::forward<Args>(args)...)) {}
    template <index_type... indices>
    Tensor_(const zipper::extents<indices...>& e) : Base(e) {}
    using Base::operator=;
};
template <typename ValueType, index_type... Indxs>
using Tensor = Tensor_<ValueType, zipper::extents<Indxs...>>;
}  // namespace zipper

#endif
