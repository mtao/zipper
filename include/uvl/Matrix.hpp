#if !defined(UVL_MATRIX_HPP)
#define UVL_MATRIX_HPP

#include "MatrixBase.hpp"
#include "storage/PlainObjectStorage.hpp"
#include "uvl/types.hpp"
#include "views/binary/AdditionView.hpp"
#include "views/unary/CastView.hpp"
#include "views/unary/ScalarProductView.hpp"
namespace uvl {

template <typename ValueType, index_type Rows, index_type Cols>
class Matrix
    : public MatrixBase<
          storage::PlainObjectStorage<ValueType, extents<Rows, Cols>>> {
   public:
    using Base =
        MatrixBase<storage::PlainObjectStorage<ValueType, extents<Rows, Cols>>>;
    using Base::view;
    using Base::view_type;

    template <concepts::MatrixBaseDerived Other>
    Matrix(const Other& other) : Base(other) {}
    template <typename... Args>
    Matrix(Args&&... args)
        requires((std::is_convertible_v<Args, index_type> && ...))
        : Base(extents<Rows, Cols>(std::forward<Args>(args)...)) {}
    template <index_type... indices>
    Matrix(const extents<indices...>& e) : Base(e) {}

    template <concepts::MatrixBaseDerived Other>
    Matrix& operator=(const Other& other) {
        view().assign(other.view());
    }
};

}  // namespace uvl

namespace uvl::views {

template <typename ValueType, index_type Rows, index_type Cols>
struct detail::ViewTraits<Matrix<ValueType, Rows, Cols>>
    : public detail::ViewTraits<
          uvl::storage::PlainObjectStorage<ValueType, extents<Rows, Cols>>> {};
}  // namespace uvl::views
#endif
