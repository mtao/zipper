#if !defined(ZIPPER_MATRIXBASE_HPP)
#define ZIPPER_MATRIXBASE_HPP

#include "ZipperBase.hpp"
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
#include "zipper/detail/constexpr_arithmetic.hpp"
#include "views/unary/IdentityView.hpp"

namespace zipper {
template <concepts::ViewDerived View>
class ArrayBase;

// template <concepts::MatrixViewDerived View>
template <concepts::ViewDerived View>
class MatrixBase : public ZipperBase<MatrixBase, View> {
   public:
    MatrixBase() = default;

    using view_type = View;
    using value_type = View::value_type;
    using extents_type = View::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;
    static_assert(extents_traits::rank == 2);
    using Base = ZipperBase<MatrixBase, View>;

    using Base::Base;
    // using Base::operator=;
    using Base::cast;
    using Base::swizzle;
    using Base::view;

    auto eval() const { return Matrix(*this); }

    template <concepts::MatrixBaseDerived Other>
    MatrixBase(const Other& other)
        requires(view_type::is_writable)
        : MatrixBase(other.view()) {}

    MatrixBase& operator=(concepts::MatrixBaseDerived auto const& v) {
        return operator=(v.view());
    }

    template <concepts::MatrixViewDerived Other>
    MatrixBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        return Base::operator=(other);
    }

    auto as_array() const {
        return ArrayBase<views::unary::IdentityView<View>>(view());
    }

    value_type trace() const { return views::reductions::Trace(view())(); }

   public:
    constexpr index_type rows() const { return extents().extent(0); }
    constexpr index_type cols() const { return extents().extent(1); }

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
        return slice<full_extent_t, Slice>(full_extent_t{}, std::forward<Slice>(s));
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
        return VectorBase<views::unary::DiagonalView<view_type, true>>(view());
    }
    auto diagonal() {
        return VectorBase<views::unary::DiagonalView<view_type, false>>(view());
    }


    template <rank_type... ranks>
    auto swizzle() const {
        return Base::template swizzle<MatrixBase, ranks...>();
    }
    auto transpose() const { return Base::template swizzle<MatrixBase, 1, 0>(); }


    template <concepts::SliceLike Slice>
    auto row_slice(const Slice& s= {}) {
        return slice(s, full_extent_t{});
    }
    template <concepts::SliceLike Slice>
    auto row_slice(const Slice& s= {}) const {
        return slice(s, full_extent_t{});
    }
    template <concepts::SliceLike Slice>
    auto col_slice(const Slice& s= {}) {
        return slice(full_extent_t{},s);
    }
    template <concepts::SliceLike Slice>
    auto col_slice(const Slice& s= {}) const {
        return slice( full_extent_t{},s);
    }

    // meh names for alignment with eigen
    template <concepts::IndexLike Index>
    auto topRows(const Index& s) {
        return row_slice(zipper::slice({},s));
    }
    template <concepts::IndexLike Index>
    auto topRows(const Index& s) const {
        return row_slice(zipper::slice({},s));
    }
    template <concepts::IndexLike Index>
    auto leftCols(const Index& s) {
        return col_slice(zipper::slice({},s));
    }
    template <concepts::IndexLike Index>
    auto leftCols(const Index& s) const {
        return col_slice(zipper::slice({},s));
    }

    template <concepts::IndexLike Index>
    auto bottomRows(const Index& s) {
        return row_slice(zipper::slice({},s));
    }
    template <concepts::IndexLike Index>
    auto bottomRows(const Index& s) const {
        return row_slice(zipper::slice({},s));
    }
    template <concepts::IndexLike Index>
    auto rightCols(const Index& s) {
        return col_slice(zipper::slice({},s));
    }
    template <concepts::IndexLike Index>
    auto rightCols(const Index& s) const {
        return col_slice(zipper::slice({},s));
    }
    template <index_type Size>
    auto topRows() {
        return row_slice(zipper::slice({},static_index<Size>{}));
    }
    template <index_type Size>
    auto topRows() const {
        return row_slice(zipper::slice({},static_index<Size>{}));
    }
    template <index_type Size>
    auto leftCols() {
        return col_slice(zipper::slice({},static_index<Size>{}));
    }
    template <index_type Size>
    auto leftCols() const {
        return col_slice(zipper::slice({},static_index<Size>{}));
    }

    template <index_type Size>
    auto bottomRows() {
        return row_slice(zipper::slice({},static_index<Size>{}));
    }
    template <index_type Size>
    auto bottomRows() const {
        return row_slice(zipper::slice({},static_index<Size>{}));
    }
    template <index_type Size>
    auto rightCols() {
        return col_slice(zipper::slice({},static_index<Size>{}));
    }
    template <index_type Size>
    auto rightCols() const {
        return col_slice(zipper::slice({},static_index<Size>{}));
    }

};

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
    using V = views::binary::MatrixProductView<typename View1::view_type,
                                               typename View2::view_type>;
    return MatrixBase<V>(V(lhs.view(), rhs.view()));
}

template <concepts::MatrixBaseDerived View>
auto operator*(View const& lhs, typename View::value_type const& rhs) {
    using V =
        views::unary::ScalarMultipliesView<typename View::value_type,
                                           typename View::view_type, true>;
    return MatrixBase<V>(V(lhs.view(), rhs));
}
template <concepts::MatrixBaseDerived View>
auto operator*(typename View::value_type const& lhs, View const& rhs) {
    using V =
        views::unary::ScalarMultipliesView<typename View::value_type,
                                           typename View::view_type, false>;
    return MatrixBase<V>(V(lhs, rhs.view()));
}

template <concepts::MatrixBaseDerived View1, concepts::VectorBaseDerived View2>
auto operator*(View1 const& lhs, View2 const& rhs) {
    using V = views::binary::MatrixVectorProductView<typename View1::view_type,
                                                     typename View2::view_type>;

    return VectorBase<V>(V(lhs.view(), rhs.view()));
}

}  // namespace zipper
#endif
