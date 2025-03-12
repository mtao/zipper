#if !defined(UVL_VECTOR_HPP)
#define UVL_VECTOR_HPP

#include "MatrixBase.hpp"
#include "storage/PlainObjectStorage.hpp"
#include "uvl/types.hpp"
namespace uvl {

template <typename ValueType, index_type Rows>
class Vector
    : public MatrixBase<
          storage::PlainObjectStorage<ValueType, extents<Rows,1>>> {
   public:
    using Base =
        MatrixBase<storage::PlainObjectStorage<ValueType, extents<Rows,1>>>;
    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;

    template <concepts::MatrixBaseDerived Other>
    Vector(const Other& other) : Base(other) {}
    template <typename... Args>
    Vector(Args&&... args)
        requires((std::is_convertible_v<Args, index_type> && ...))
        : Base(extents<Rows,1>(std::forward<Args>(args)...)) {}
    template <index_type... indices>
    Vector(const extents<indices...>& e) : Base(e) {}

    template <concepts::MatrixBaseDerived Other>
    Vector& operator=(const Other& other) {
        view().assign(other.view());
    }

    
    template <typename... Args>
    const value_type& operator()(Args&&... idxs) const

    {
        return view()(std::forward<Args>(idxs)...);
    }
    template <typename... Args>
    value_type& operator()(Args&&... idxs)

    {
        return view()(std::forward<Args>(idxs)...);
    }

};

}  // namespace uvl

namespace uvl::views {

template <typename ValueType, index_type Rows>
struct detail::ViewTraits<Vector<ValueType, Rows>>
    : public detail::ViewTraits<
          uvl::storage::PlainObjectStorage<ValueType, extents<Rows>>> {};
}  // namespace uvl::views
#endif
