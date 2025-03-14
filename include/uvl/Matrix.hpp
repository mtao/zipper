#if !defined(UVL_MATRIX_HPP)
#define UVL_MATRIX_HPP

#include "MatrixBase.hpp"
#include "concepts/MatrixBaseDerived.hpp"
#include "storage/PlainObjectStorage.hpp"
#include "uvl/types.hpp"
namespace uvl {

template <typename ValueType, index_type Rows, index_type Cols>
class Matrix
    : public MatrixBase<
          storage::PlainObjectStorage<ValueType, uvl::extents<Rows, Cols>>> {
   public:
    using Base = MatrixBase<
        storage::PlainObjectStorage<ValueType, uvl::extents<Rows, Cols>>>;
    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using Base::extent;
    using Base::extents;

    template <concepts::MatrixViewDerived Other>
    Matrix(const Other& other) : Base(other) {}
    template <concepts::MatrixBaseDerived Other>
    Matrix(const Other& other) : Base(other) {}
    template <typename... Args>
    Matrix(Args&&... args)
        requires((std::is_convertible_v<Args, index_type> && ...))
        : Base(uvl::extents<Rows, Cols>(std::forward<Args>(args)...)) {}
    template <index_type... indices>
    Matrix(const uvl::extents<indices...>& e) : Base(e) {}
    using Base::operator=;

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
template <concepts::MatrixBaseDerived MB>
Matrix(const MB& o)
    -> Matrix<typename MB::value_type, MB::extents_type::static_extent(0),
              MB::extents_type::static_extent(1)>;

}  // namespace uvl

namespace uvl::views {

template <typename ValueType, index_type Rows, index_type Cols>
struct detail::ViewTraits<Matrix<ValueType, Rows, Cols>>
    : public detail::ViewTraits<uvl::storage::PlainObjectStorage<
          ValueType, uvl::extents<Rows, Cols>>> {};
}  // namespace uvl::views
#endif
