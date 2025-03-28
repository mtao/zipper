#if !defined(ZIPPER_VECTOR_HPP)
#define ZIPPER_VECTOR_HPP

#include "VectorBase.hpp"
#include "storage/PlainObjectStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, index_type Rows>
class Vector : public VectorBase<
                   storage::PlainObjectStorage<ValueType, zipper::extents<Rows>>> {
   public:
    using Base =
        VectorBase<storage::PlainObjectStorage<ValueType, zipper::extents<Rows>>>;
    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using Base::extent;
    using Base::extents;

    template <concepts::VectorViewDerived Other>
    Vector(const Other& other) : Base(other) {}
    template <concepts::VectorBaseDerived Other>
    Vector(const Other& other) : Base(other) {}

    template <concepts::ViewDerived Other>
    Vector(const Other& other) : Base(other) {}
    template <typename... Args>
    Vector(Args&&... args)
        requires((std::is_convertible_v<Args, index_type> && ...))
        : Base(zipper::extents<Rows>(std::forward<Args>(args)...)) {}
    template <index_type... indices>
    Vector(const zipper::extents<indices...>& e) : Base(e) {}
    using Base::operator=;

};

template <concepts::VectorViewDerived MB>
Vector(const MB& o)
    -> Vector<typename MB::value_type, MB::extents_type::static_extent(0)>;
template <concepts::VectorBaseDerived MB>
Vector(const MB& o)
    -> Vector<typename MB::value_type, MB::extents_type::static_extent(0)>;
}  // namespace zipper

namespace zipper::views {

template <typename ValueType, index_type Rows>
struct detail::ViewTraits<Vector<ValueType, Rows>>
    : public detail::ViewTraits<
          zipper::storage::PlainObjectStorage<ValueType, extents<Rows>>> {};
}  // namespace zipper::views
#endif
