#if !defined(ZIPPER_FORM_HPP)
#define ZIPPER_FORM_HPP

#include "FormBase.hpp"
#include "concepts/FormBaseDerived.hpp"
#include "storage/PlainObjectStorage.hpp"
#include "storage/SpanStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, concepts::ExtentsType Extents, bool LeftMajor>
class Form_ : public FormBase<storage::PlainObjectStorage<
                  ValueType, Extents, storage::tensor_layout<LeftMajor>>> {
   public:
    using layout_type = storage::tensor_layout<LeftMajor>;
    using Base =
        FormBase<storage::PlainObjectStorage<ValueType, Extents, layout_type>>;
    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using span_type =
        FormBase<storage::SpanStorage<ValueType, Extents, layout_type>>;
    using const_span_type =
        FormBase<storage::SpanStorage<const ValueType, Extents, layout_type>>;

    using Base::extent;
    using Base::Base;
    using Base::extents;

    Form_(const Form_& o) = default;
    Form_(Form_&& o) = default;
    Form_& operator=(const Form_& o) = default;
    template <concepts::ViewDerived Other>
    Form_(const Other& other) : Base(other) {}
    template <concepts::FormBaseDerived Other>
    Form_(const Other& other) : Base(other) {}
    template <concepts::IndexLike... Args>
    Form_(Args&&... args)
        : Base(Extents(std::forward<Args>(args)...)) {}
    template <index_type... indices>
    Form_(const zipper::extents<indices...>& e) : Base(e) {}
    Form_& operator=(Form_&& o) {
        view().operator=(std::move(o.view()));
        return *this;
    }
    using Base::operator=;
};
template <typename ValueType, index_type... Indxs>
using Form = Form_<ValueType, zipper::extents<Indxs...>>;
}  // namespace zipper

#endif
