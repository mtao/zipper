#if !defined(ZIPPER_MATRIX_HPP)
#define ZIPPER_MATRIX_HPP

#include "MatrixBase.hpp"
#include "concepts/MatrixBaseDerived.hpp"
#include "storage/PlainObjectStorage.hpp"
#include "storage/SpanStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, index_type Rows, index_type Cols, bool RowMajor>
class Matrix
    : public MatrixBase<
          storage::PlainObjectStorage<ValueType, zipper::extents<Rows, Cols>,
                                      storage::matrix_layout<RowMajor>>> {
   public:
    using layout_type = storage::matrix_layout<RowMajor>;
    using Base = MatrixBase<storage::PlainObjectStorage<
        ValueType, zipper::extents<Rows, Cols>, layout_type>>;

    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using Base::col;
    using Base::extent;
    using Base::extents;
    using Base::row;
    using extents_traits = detail::ExtentsTraits<extents_type>;
    using span_type =
        MatrixBase<storage::SpanStorage<ValueType, zipper::extents<Rows, Cols>,
                                        layout_type>>;
    using const_span_type =
        MatrixBase<storage::SpanStorage<const ValueType, zipper::extents<Rows, Cols>,
                                        layout_type>>;
    using Base::transpose;

    Matrix()
        requires(extents_traits::is_static)
    = default;

    Matrix(index_type dyn_size)
        requires(extents_traits::rank_dynamic == 1)
        : Base(extents_type(dyn_size)) {}

    Matrix(index_type rows, index_type cols)
        requires(extents_traits::is_dynamic)
        : Base(extents_type(rows, cols)) {}

    Matrix(const extents_type& e)
        requires(extents_traits::is_dynamic)
        : Base(e) {}
    Matrix(const extents_type&)
        requires(extents_traits::is_static)
        : Base() {}

#if defined(NDEBUG)
    Matrix(index_type , index_type )
#else
    Matrix(index_type rows, index_type cols)
#endif
        requires(extents_traits::is_static)
        : Base() {
            assert(rows == extent(0));
            assert(cols == extent(1));
        }

    template <concepts::MatrixBaseDerived Other>
    Matrix(const Other& other) : Base(other) {}

    template <concepts::ViewDerived Other>
    Matrix(const Other& other) : Base(other) {}

    Matrix(const Matrix& other) : Base(other.view()) {}
    template <index_type R2, index_type C2>
    Matrix(const Matrix<value_type, R2, C2>& other) : Base(other.view()) {}
    // template <concepts::MatrixViewDerived Other>
    // Matrix(const Other& other) : Base(other) {}
    // template <concepts::MatrixBaseDerived Other>
    // Matrix(const Other& other) : Base(other) {}
    // template <concepts::ViewDerived Other>
    // Matrix(const Other& other) : Base(other) {}
    // template <typename... Args>
    // Matrix(Args&&... args)
    //     requires(concepts::IndexPackLike<Args...>)
    //     : Base(zipper::extents<Rows, Cols>(std::forward<Args>(args)...)) {}
    // template <index_type... indices>
    // Matrix(const zipper::extents<indices...>& e) : Base(e) {}
    using Base::operator=;
    Matrix& operator=(Matrix&& other) {
        Base::operator=(std::move(other.view()));
        return *this;
    }

    Matrix& operator=(const Matrix& other) {
        Base::operator=(other.view());
        return *this;
    }
    template <index_type R2, index_type C2>
    Matrix& operator=(const Matrix<value_type, R2, C2>& other) {
        Base::operator=(other.view());
        return *this;
    }

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
    -> Matrix<std::decay_t<typename MB::value_type>, MB::extents_type::static_extent(0),
              MB::extents_type::static_extent(1)>;

template <concepts::MatrixBaseDerived MB>
Matrix(const MB& o)
    -> Matrix<std::decay_t<typename MB::value_type>, MB::extents_type::static_extent(0),
              MB::extents_type::static_extent(1)>;

}  // namespace zipper

namespace zipper::views {

template <typename ValueType, index_type Rows, index_type Cols, bool RowMajor>
struct detail::ViewTraits<Matrix<ValueType, Rows, Cols, RowMajor>>
    : public detail::ViewTraits<zipper::storage::PlainObjectStorage<
          ValueType, zipper::extents<Rows, Cols>,
          std::conditional_t<RowMajor, std::experimental::layout_left,
                             std::experimental::layout_right>>> {};
}  // namespace zipper::views
#endif
