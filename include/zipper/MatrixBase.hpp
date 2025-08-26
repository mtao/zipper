#include "zipper/concepts/IndexLike.hpp"
#if !defined(ZIPPER_MATRIXBASE_HPP)
#define ZIPPER_MATRIXBASE_HPP

#include "ZipperBase.hpp"
#include "as.hpp"
#include "concepts/MatrixBaseDerived.hpp"
#include "concepts/MatrixViewDerived.hpp"
#include "concepts/VectorBaseDerived.hpp"
#include "zipper/types.hpp"
//
#include "views/binary/MatrixProductView.hpp"
#include "views/binary/MatrixVectorProductView.hpp"
#include "views/reductions/Trace.hpp"
// #include "views/reductions/Determinant.hpp"
#include "ArrayBase.hpp"
#include "views/unary/IdentityView.hpp"
#include "zipper/detail/PartialReductionDispatcher.hpp"
#include "zipper/detail/constexpr_arithmetic.hpp"
#include "zipper/detail/extents/get_extent.hpp"

namespace zipper {
template <concepts::QualifiedViewDerived View>
class ArrayBase;

// template <concepts::MatrixViewDerived View>
template <concepts::QualifiedViewDerived View>
class MatrixBase : public ZipperBase<MatrixBase, View> {
   public:
    MatrixBase() = default;

    using view_type = View;
    using view_traits = views::detail::ViewTraits<View>;
    using value_type = view_traits::value_type;
    using extents_type = view_traits::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;
    static_assert(extents_traits::rank == 2);
    using Base = ZipperBase<MatrixBase, View>;

    using Base::Base;
    // using Base::operator=;
    using Base::cast;
    using Base::extent;
    using Base::extents;
    using Base::swizzle;
    using Base::view;

    auto eval() const { return Matrix(*this); }

    template <concepts::MatrixBaseDerived Other>
    MatrixBase(const Other& other)
        requires(view_traits::is_writable)
        : MatrixBase(other.view()) {}

    MatrixBase& operator=(concepts::MatrixBaseDerived auto const& v) {
        return operator=(v.view());
    }

    template <concepts::MatrixViewDerived Other>
    MatrixBase& operator=(const Other& other)
        requires(view_traits::is_writable)
    {
        return Base::operator=(other);
    }

    void resize(index_type rows, index_type cols) {
        constexpr static bool dynamic_row =
            extents_traits::is_dynamic_extent(0);
        constexpr static bool dynamic_col =
            extents_traits::is_dynamic_extent(1);
        if constexpr (dynamic_row && dynamic_col) {
            view().resize(extents_type{rows, cols});
        } else if constexpr (dynamic_row) {
            view().resize(extents_type{rows});
        } else if constexpr (dynamic_col) {
            view().resize(extents_type{cols});
        }
    }
    void resize(index_type size)
        requires(extents_traits::rank_dynamic == 1)
    {
        view().resize(extents_type{size});
    }

    auto as_array() const { return zipper::as_array(*this); }

    value_type trace() const { return views::reductions::Trace(view())(); }

    template <typename... Args>
    auto operator()(Args&&... idxs) const -> decltype(auto)

    {
        decltype(auto) r = Base::operator()(std::forward<Args>(idxs)...);
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
        decltype(auto) r = Base::operator()(std::forward<Args>(idxs)...);
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

   public:
    constexpr index_type rows() const { return extent(0); }
    constexpr index_type cols() const { return extent(1); }

    template <typename Slice>
    auto row() {
        return slice<Slice, full_extent_t>();
    }
    template <typename Slice>
    auto col() {
        return slice<full_extent_t, Slice>();
    }

    template <typename Slice>
    auto row() const {
        return slice<Slice, full_extent_t>();
    }
    template <typename Slice>
    auto col() const {
        return slice<full_extent_t, Slice>();
    }

    template <typename Slice>
    auto row(Slice&& s) {
        return slice<Slice, full_extent_t>(std::forward<Slice>(s),
                                           full_extent_t{});
    }
    template <typename Slice>
    auto col(Slice&& s) {
        return slice<full_extent_t, Slice>(full_extent_t{},
                                           std::forward<Slice>(s));
    }

    template <typename Slice>
    auto row(Slice&& s) const {
        return slice<Slice, full_extent_t>(std::forward<Slice>(s),
                                           full_extent_t{});
    }
    template <typename Slice>
    auto col(Slice&& s) const {
        return slice<full_extent_t, Slice>(full_extent_t{},
                                           std::forward<Slice>(s));
    }
    template <typename... Slices>
    auto slice() {
        auto v = Base::template slice_view<Slices...>();
        using V = std::decay_t<decltype(v)>;
        if constexpr (V::extents_type::rank() == 2) {
            return MatrixBase<V>(std::move(v));
        } else {
            static_assert(V::extents_type::rank() == 1);
            return VectorBase<V>(std::move(v));
        }
    }

    template <typename... Slices>
    auto slice(Slices&&... slices) const {
        auto v = Base::template slice_view<Slices...>(
            std::forward<Slices>(slices)...);
        using V = std::decay_t<decltype(v)>;
        if constexpr (V::extents_type::rank() == 2) {
            return MatrixBase<V>(std::move(v));
        } else {
            static_assert(V::extents_type::rank() == 1);
            return VectorBase<V>(std::move(v));
        }
    }
    template <typename... Slices>
    auto slice() const {
        auto v = Base::template slice_view<Slices...>();
        using V = std::decay_t<decltype(v)>;
        if constexpr (V::extents_type::rank() == 2) {
            return MatrixBase<V>(std::move(v));
        } else {
            static_assert(V::extents_type::rank() == 1);
            return VectorBase<V>(std::move(v));
        }
    }

    template <typename... Slices>
    auto slice(Slices&&... slices) {
        auto v = Base::template slice_view<Slices...>(
            std::forward<Slices>(slices)...);
        using V = std::decay_t<decltype(v)>;
        if constexpr (V::extents_type::rank() == 2) {
            return MatrixBase<V>(std::move(v));
        } else {
            static_assert(V::extents_type::rank() == 1);
            return VectorBase<V>(std::move(v));
        }
    }

    auto diagonal() const {
        return VectorBase<views::unary::DiagonalView<const view_type>>(view());
    }
    auto diagonal() {
        return VectorBase<views::unary::DiagonalView<view_type>>(view());
    }

    template <rank_type... ranks>
    auto swizzle() const {
        return Base::template swizzle<MatrixBase, ranks...>();
    }
    auto transpose() const {
        return Base::template swizzle<MatrixBase, 1, 0>();
    }

    template <concepts::SliceLike Slice>
    auto row_slice(const Slice& s = {}) {
        return slice(s, full_extent_t{});
    }
    template <concepts::SliceLike Slice>
    auto row_slice(const Slice& s = {}) const {
        return slice(s, full_extent_t{});
    }
    template <concepts::SliceLike Slice>
    auto col_slice(const Slice& s = {}) {
        return slice(full_extent_t{}, s);
    }
    template <concepts::SliceLike Slice>
    auto col_slice(const Slice& s = {}) const {
        return slice(full_extent_t{}, s);
    }

    // meh names for alignment with eigen
    template <concepts::IndexLike Index>
    auto topRows(const Index& s) {
        return row_slice(zipper::slice({}, s));
    }
    template <concepts::IndexLike Index>
    auto topRows(const Index& s) const {
        return row_slice(zipper::slice({}, s));
    }
    template <concepts::IndexLike Index>
    auto leftCols(const Index& s) {
        return col_slice(zipper::slice({}, s));
    }
    template <concepts::IndexLike Index>
    auto leftCols(const Index& s) const {
        return col_slice(zipper::slice({}, s));
    }

    template <concepts::IndexLike Index>
    auto bottomRows(const Index& s) {
        auto end = detail::extents::get_extent<0>(extents());
        detail::ConstexprArithmetic size(s);
        return row_slice(zipper::slice((end - size).value(), s));
    }
    template <concepts::IndexLike Index>
    auto bottomRows(const Index& s) const {
        auto end = detail::extents::get_extent<0>(extents());
        detail::ConstexprArithmetic size(s);
        return row_slice(zipper::slice((end - size).value(), s));
    }
    template <concepts::IndexLike Index>
    auto rightCols(const Index& s) {
        auto end = detail::extents::get_extent<1>(extents());
        detail::ConstexprArithmetic size(s);
        return col_slice(zipper::slice((end - size).value(), s));
    }
    template <concepts::IndexLike Index>
    auto rightCols(const Index& s) const {
        auto end = detail::extents::get_extent<1>(extents());
        detail::ConstexprArithmetic size(s);
        return col_slice(zipper::slice((end - size).value(), s));
    }
    template <index_type Size>
    auto topRows() {
        return row_slice(zipper::slice({}, static_index_t<Size>{}));
    }
    template <index_type Size>
    auto topRows() const {
        return row_slice(zipper::slice({}, static_index_t<Size>{}));
    }
    template <index_type Size>
    auto leftCols() {
        return col_slice(zipper::slice({}, static_index_t<Size>{}));
    }
    template <index_type Size>
    auto leftCols() const {
        return col_slice(zipper::slice({}, static_index_t<Size>{}));
    }

    template <index_type Size>
    auto bottomRows() {
        detail::ConstexprArithmetic size(static_index_t<Size>{});
        auto t = size - detail::extents::get_extent<0>(extents());
        return row_slice(zipper::slice(t.value(), size.value()));
    }
    template <index_type Size>
    auto bottomRows() const {
        detail::ConstexprArithmetic size(static_index_t<Size>{});
        auto t = size - detail::extents::get_extent<0>(extents());
        return row_slice(zipper::slice(t.value(), size.value()));
    }
    template <index_type Size>
    auto rightCols() {
        detail::ConstexprArithmetic size(static_index_t<Size>{});
        auto t = size - detail::extents::get_extent<1>(extents());
        return col_slice(zipper::slice(t.value(), size.value()));
    }
    template <index_type Size>
    auto rightCols() const {
        detail::ConstexprArithmetic size(static_index_t<Size>{});
        auto t = size - detail::extents::get_extent<1>(extents());
        return col_slice(zipper::slice(t.value(), size.value()));
    }

    auto rowwise() {
        // we're reducing the first cols
        return detail::PartialReductionDispatcher<VectorBase, view_type, 1>(
            view());
    }
    auto colwise() {
        // we're reducing the first rows
        return detail::PartialReductionDispatcher<VectorBase, view_type, 0>(
            view());
    }
    auto rowwise() const {
        // we're reducing the first cols
        return detail::PartialReductionDispatcher<VectorBase, const view_type,
                                                  1>(view());
    }
    auto colwise() const {
        // we're reducing the first rows
        return detail::PartialReductionDispatcher<VectorBase, const view_type,
                                                  0>(view());
    }
};

template <typename T>
using MatrixXX = Matrix<T, dynamic_extent, dynamic_extent>;
template <typename T, index_type R>
using MatrixRX = Matrix<T, R, dynamic_extent>;
template <typename T, index_type C>
using MatrixXC = Matrix<T, dynamic_extent, C>;

template <concepts::MatrixViewDerived View>
MatrixBase(View&& view) -> MatrixBase<View>;
template <concepts::MatrixViewDerived View>
MatrixBase(const View& view) -> MatrixBase<View>;

UNARY_DECLARATION(MatrixBase, LogicalNot, operator!)
UNARY_DECLARATION(MatrixBase, BitNot, operator~)
UNARY_DECLARATION(MatrixBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(MatrixBase, Divides, operator/)

BINARY_DECLARATION(MatrixBase, Plus, operator+)
BINARY_DECLARATION(MatrixBase, Minus, operator-)
//

template <concepts::MatrixBaseDerived View1, concepts::MatrixBaseDerived View2>
auto operator==(View1 const& lhs, View2 const& rhs) {
    return (lhs.as_array() == rhs.as_array()).all();
}
template <concepts::MatrixBaseDerived View1, concepts::MatrixBaseDerived View2>
auto operator!=(View1 const& lhs, View2 const& rhs) {
    return (lhs.as_array() == rhs.as_array()).all();
}

template <concepts::MatrixBaseDerived View1, concepts::MatrixBaseDerived View2>
auto operator*(View1 const& lhs, View2 const& rhs) {
    using V = views::binary::MatrixProductView<const typename View1::view_type,
                                               const typename View2::view_type>;
    return MatrixBase<V>(V(lhs.view(), rhs.view()));
}

template <concepts::MatrixBaseDerived View>
auto operator*(View const& lhs, typename View::value_type const& rhs) {
    using V = views::unary::ScalarMultipliesView<
        typename View::value_type, const typename View::view_type, true>;
    return MatrixBase<V>(V(lhs.view(), rhs));
}
template <concepts::MatrixBaseDerived View>
auto operator*(typename View::value_type const& lhs, View const& rhs) {
    using V = views::unary::ScalarMultipliesView<
        typename View::value_type, const typename View::view_type, false>;
    return MatrixBase<V>(V(lhs, rhs.view()));
}

template <concepts::MatrixBaseDerived View1, concepts::VectorBaseDerived View2>
auto operator*(View1 const& lhs, View2 const& rhs) {
    using V =
        views::binary::MatrixVectorProductView<const typename View1::view_type,
                                               const typename View2::view_type>;

    return VectorBase<V>(V(lhs.view(), rhs.view()));
}

}  // namespace zipper
#endif
