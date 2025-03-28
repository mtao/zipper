#if !defined(ZIPPER_FORM_HPP)
#define ZIPPER_FORM_HPP

#include "FormBase.hpp"
#include "concepts/FormBaseDerived.hpp"
#include "storage/PlainObjectStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, index_type... Dims>
class Form
    : public FormBase<
          storage::PlainObjectStorage<ValueType, zipper::extents<Dims...>>> {
   public:
    using Base = FormBase<
        storage::PlainObjectStorage<ValueType, zipper::extents<Dims...>>>;
    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using Base::extent;
    using Base::extents;

    template <concepts::ViewDerived Other>
    Form(const Other& other) : Base(other) {}
    template <concepts::FormBaseDerived Other>
    Form(const Other& other) : Base(other) {}
    template <typename... Args>
    Form(Args&&... args)
        requires((std::is_convertible_v<Args, index_type> && ...))
        : Base(zipper::extents<Dims...>(std::forward<Args>(args)...)) {}
    template <index_type... indices>
    Form(const zipper::extents<indices...>& e) : Base(e) {}
    using Base::operator=;

};
}  // namespace zipper

namespace zipper::views {

template <typename ValueType, index_type... Dims>
struct detail::ViewTraits<Form<ValueType, Dims...>>
    : public detail::ViewTraits<
          zipper::storage::PlainObjectStorage<ValueType, zipper::extents<Dims...>>> {
};
}  // namespace zipper::views
#endif
