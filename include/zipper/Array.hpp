#if !defined(ZIPPER_ARRAY_HPP)
#define ZIPPER_ARRAY_HPP

#include "ArrayBase.hpp"
#include "concepts/ArrayBaseDerived.hpp"
#include "concepts/ExtentsType.hpp"
#include "concepts/MatrixViewDerived.hpp"
#include "concepts/VectorViewDerived.hpp"
#include "storage/PlainObjectStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, concepts::ExtentsType Extents, bool LeftMajor>
class Array_
    : public ArrayBase<storage::PlainObjectStorage<ValueType, Extents>> {
   public:
    using Base = ArrayBase<storage::PlainObjectStorage<ValueType, Extents>>;
    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using Base::extent;
    using Base::extents;

    Array_(const Array_& o) = default;
    Array_(Array_&& o) = default;
    Array_& operator=(const Array_& o) = default;
    Array_& operator=(Array_&& o) = default;
    template <concepts::ViewDerived Other>
    Array_(const Other& other) : Base(other) {}
    template <concepts::ArrayBaseDerived Other>
    Array_(const Other& other) : Base(other) {}
    template <typename... Args>
    Array_(Args&&... args)
        requires((std::is_convertible_v<Args, index_type> && ...))
        : Base(Extents(std::forward<Args>(args)...)) {}
    template <index_type... indices>
    Array_(const zipper::extents<indices...>& e) : Base(e) {}
};
// template <concepts::ViewDerived MB>
// Array(const MB& o)->typename MB::array_typeA;
//  template <concepts::ViewDerived MB>
//  Array(const MB& o)
//      ->typename detail::get_array_type<typename MB::value_type,
//                                        typename MB::extents_type>::value;

template <concepts::MatrixViewDerived MB>
Array_(const MB& o)
    -> Array_<typename MB::value_type, typename MB::extents_type>;

template <concepts::VectorViewDerived MB>
Array_(const MB& o)
    -> Array_<typename MB::value_type, typename MB::extents_type>;

template <typename ValueType, index_type... Indxs>
using Array = Array_<ValueType, zipper::extents<Indxs...>>;
}  // namespace zipper
#endif
