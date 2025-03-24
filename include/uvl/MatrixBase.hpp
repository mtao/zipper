#if !defined(UVL_MATRIXBASE_HPP)
#define UVL_MATRIXBASE_HPP

#include "UVLBase.hpp"
#include "concepts/MatrixBaseDerived.hpp"
#include "concepts/MatrixViewDerived.hpp"
#include "concepts/VectorBaseDerived.hpp"
#include "uvl/types.hpp"
//
#include "views/binary/MatrixProductView.hpp"
#include "views/binary/MatrixVectorProductView.hpp"
#include "views/reductions/Trace.hpp"
// #include "views/reductions/Determinant.hpp"
#include "ArrayBase.hpp"
#include "views/unary/IdentityView.hpp"

namespace uvl {
template <concepts::ViewDerived View>
class ArrayBase;

// template <concepts::MatrixViewDerived View>
template <concepts::ViewDerived View>
class MatrixBase : public UVLBase<MatrixBase, View> {
   public:
    MatrixBase() = default;

    using view_type = View;
    using value_type = View::value_type;
    using extents_type = View::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;
    static_assert(extents_traits::rank == 2);
    using Base = UVLBase<MatrixBase, View>;

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
        return slice<full_extent_t, Slice>(full_extent, std::forward<Slice>(s));
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
    auto transpose() const { return swizzle<MatrixBase, 1, 0>(); }

    template <rank_type... ranks>
    auto swizzle() const {
        return Base::template swizzle<MatrixBase, ranks...>();
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
    return MatrixBase<views::binary::MatrixProductView<
        typename View1::view_type, typename View2::view_type>>(lhs.view(),
                                                               rhs.view());
}

template <concepts::MatrixBaseDerived View>
auto operator*(View const& lhs, typename View::value_type const& rhs) {
    return MatrixBase<views::unary::ScalarMultipliesView<
        typename View::value_type, typename View::view_type, true>>(lhs.view(), rhs);
}
template <concepts::MatrixBaseDerived View>
auto operator*(typename View::value_type const& lhs, View const& rhs) {
    return MatrixBase<views::unary::ScalarMultipliesView<
        typename View::value_type,typename View::view_type, false>>(lhs, rhs.view());
}

template <concepts::MatrixBaseDerived View1, concepts::VectorBaseDerived View2>
auto operator*(View1 const& lhs, View2 const& rhs) {
    return VectorBase<views::binary::MatrixVectorProductView<
        typename View1::view_type, typename View2::view_type>>(lhs.view(),
                                                               rhs.view());
}

}  // namespace uvl
#endif
