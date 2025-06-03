#if !defined(ZIPPER_FORM_HPP)
#define ZIPPER_FORM_HPP

#include "FormBase.hpp"
#include "FormSpan.hpp"
#include "concepts/FormBaseDerived.hpp"
#include "storage/PlainObjectStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, concepts::ExtentsType Extents, bool LeftMajor>
class Form_ : public FormBase<storage::PlainObjectStorage<ValueType, Extents>> {
   public:
    using Base = FormBase<storage::PlainObjectStorage<ValueType, Extents>>;
    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using span_type = FormSpan_<ValueType, Extents>;
    using Base::extent;
    using Base::extents;

    Form_(const Form_& o) = default;
    Form_(Form_&& o) = default;
    Form_& operator=(const Form_& o) = default;
    Form_& operator=(Form_&& o) = default;
    template <concepts::ViewDerived Other>
    Form_(const Other& other) : Base(other) {}
    template <concepts::FormBaseDerived Other>
    Form_(const Other& other) : Base(other) {}
    template <typename... Args>
    Form_(Args&&... args)
        requires((std::is_convertible_v<Args, index_type> && ...))
        : Base(Extents(std::forward<Args>(args)...)) {}
    template <index_type... indices>
    Form_(const zipper::extents<indices...>& e) : Base(e) {}
    using Base::operator=;
};
template <typename ValueType, index_type... Indxs>
using Form = Form_<ValueType, zipper::extents<Indxs...>>;
}  // namespace zipper

#endif
