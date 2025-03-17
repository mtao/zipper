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
    using Base::col;
    using Base::extent;
    using Base::extents;
    using Base::row;

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
    auto operator()(Args&&... idxs) const -> decltype(auto)

    {
        decltype(auto) r = view()(std::forward<Args>(idxs)...);
        if constexpr (std::is_same_v<std::decay_t<decltype(r)>, value_type>) {
            return r;
        } else {
            using R = typename std::decay_t<decltype(r)>;
            if constexpr (R::extents_type::rank() == 1) {
                return VectorBase<R>(r);
            } else if constexpr (R::extents_type::rank() == 2) {
                return MatrixBase<R>(r);
            }
        }
    }
    template <typename... Args>
    auto operator()(Args&&... idxs) -> decltype(auto)

    {
        decltype(auto) r = view()(std::forward<Args>(idxs)...);
        if constexpr (std::is_same_v<std::decay_t<decltype(r)>, value_type>) {
            return r;
        } else {
            using R = typename std::decay_t<decltype(r)>;
            if constexpr (R::extents_type::rank() == 1) {
                return VectorBase<R>(r);
            } else if constexpr (R::extents_type::rank() == 2) {
                return MatrixBase<R>(r);
            }
        }
    }
};
template <concepts::MatrixViewDerived MB>
Matrix(const MB& o)
    -> Matrix<typename MB::value_type, MB::extents_type::static_extent(0),
              MB::extents_type::static_extent(1)>;

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
