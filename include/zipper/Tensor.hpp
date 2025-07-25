#if !defined(ZIPPER_TENSOR_HPP)
#define ZIPPER_TENSOR_HPP

#include "TensorBase.hpp"
#include "concepts/ExtentsType.hpp"
#include "concepts/TensorBaseDerived.hpp"
#include "storage/PlainObjectStorage.hpp"
#include "storage/SpanStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, concepts::ExtentsType Extents, bool LeftMajor>
class Tensor_ : public TensorBase<storage::PlainObjectStorage<
                    ValueType, Extents, storage::tensor_layout<LeftMajor>>> {
   public:
    using layout_type = storage::tensor_layout<LeftMajor>;
    using Base = TensorBase<
        storage::PlainObjectStorage<ValueType, Extents, layout_type>>;
    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using Base::extent;
    using Base::extents;
    using span_type =
        TensorBase<storage::SpanStorage<ValueType, Extents, layout_type>>;
    using const_span_type =
        TensorBase<storage::SpanStorage<const ValueType, Extents, layout_type>>;

    template <concepts::ViewDerived Other>
    Tensor_(const Other& other) : Base(other) {}
    template <concepts::TensorBaseDerived Other>
    Tensor_(const Other& other) : Base(other) {}
    template <concepts::IndexLike... Args>
    Tensor_(Args&&... args)
        : Base(Extents(std::forward<Args>(args)...)) {}
    Tensor_() = default;
    Tensor_(const Tensor_&) = default;
    Tensor_(Tensor_&&) = default;
    template <index_type... indices>
    Tensor_(const zipper::extents<indices...>& e) : Base(e) {}
    Tensor_& operator=(Tensor_&& o) {
        view().operator=(std::move(o.view()));
        return *this;
    }
    using Base::operator=;
};
template <typename ValueType, index_type... Indxs>
using Tensor = Tensor_<ValueType, zipper::extents<Indxs...>>;
}  // namespace zipper

#endif
