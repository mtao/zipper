#if !defined(ZIPPER_ARRAY_HPP)
#define ZIPPER_ARRAY_HPP

#include "ArrayBase.hpp"
#include "concepts/ArrayBaseDerived.hpp"
#include "concepts/MatrixViewDerived.hpp"
#include "concepts/VectorViewDerived.hpp"
#include "storage/PlainObjectStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, index_type... Dims>
class Array
    : public ArrayBase<
          storage::PlainObjectStorage<ValueType, zipper::extents<Dims...>>> {
   public:
    using Base = ArrayBase<
        storage::PlainObjectStorage<ValueType, zipper::extents<Dims...>>>;
    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using Base::extent;
    using Base::extents;

    template <concepts::ViewDerived Other>
    Array(const Other& other) : Base(other) {}
    template <concepts::ArrayBaseDerived Other>
    Array(const Other& other) : Base(other) {}
    template <typename... Args>
    Array(Args&&... args)
        requires((std::is_convertible_v<Args, index_type> && ...))
        : Base(zipper::extents<Dims...>(std::forward<Args>(args)...)) {}
    template <index_type... indices>
    Array(const zipper::extents<indices...>& e) : Base(e) {}
    using Base::operator=;

};
// template <concepts::ViewDerived MB>
// Array(const MB& o)->typename MB::array_typeA;
//  template <concepts::ViewDerived MB>
//  Array(const MB& o)
//      ->typename detail::get_array_type<typename MB::value_type,
//                                        typename MB::extents_type>::value;

template <concepts::MatrixViewDerived MB>
Array(const MB& o)
    -> Array<typename MB::value_type, MB::extents_type::static_extent(0),
             MB::extents_type::static_extent(1)>;

template <concepts::VectorViewDerived MB>
Array(const MB& o)
    -> Array<typename MB::value_type, MB::extents_type::static_extent(0)>;

}  // namespace zipper

namespace zipper::views {

template <typename ValueType, index_type... Dims>
struct detail::ViewTraits<Array<ValueType, Dims...>>
    : public detail::ViewTraits<
          zipper::storage::PlainObjectStorage<ValueType, zipper::extents<Dims...>>> {
};
}  // namespace zipper::views
#endif
