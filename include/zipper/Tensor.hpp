#if !defined(ZIPPER_TENSOR_HPP)
#define ZIPPER_TENSOR_HPP

#include "TensorBase.hpp"
#include "TensorSpan.hpp"
#include "concepts/TensorBaseDerived.hpp"
#include "storage/PlainObjectStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, index_type... Dims>
class Tensor
    : public TensorBase<
          storage::PlainObjectStorage<ValueType, zipper::extents<Dims...>>> {
   public:
    using Base = TensorBase<
        storage::PlainObjectStorage<ValueType, zipper::extents<Dims...>>>;
    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using Base::extent;
    using Base::extents;
    using span_type = TensorSpan<ValueType, false, Dims...>;

    template <concepts::ViewDerived Other>
    Tensor(const Other& other) : Base(other) {}
    template <concepts::TensorBaseDerived Other>
    Tensor(const Other& other) : Base(other) {}
    template <typename... Args>
    Tensor(Args&&... args)
        requires((std::is_convertible_v<Args, index_type> && ...))
        : Base(zipper::extents<Dims...>(std::forward<Args>(args)...)) {}
    template <index_type... indices>
    Tensor(const zipper::extents<indices...>& e) : Base(e) {}
    using Base::operator=;

};
}  // namespace zipper

namespace zipper::views {

template <typename ValueType, index_type... Dims>
struct detail::ViewTraits<Tensor<ValueType, Dims...>>
    : public detail::ViewTraits<
          zipper::storage::PlainObjectStorage<ValueType, zipper::extents<Dims...>>> {
};
}  // namespace zipper::views
#endif
