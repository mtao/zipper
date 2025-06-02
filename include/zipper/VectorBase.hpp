
#if !defined(ZIPPER_VECTORBASE_HPP)
#define ZIPPER_VECTORBASE_HPP

#include "ZipperBase.hpp"
#include "as.hpp"
#include "concepts/VectorBaseDerived.hpp"
#include "concepts/VectorViewDerived.hpp"
//
#include "ArrayBase.hpp"
#include "FormBase.hpp"
#include "MatrixBase.hpp"
#include "TensorBase.hpp"
#include "views/binary/CrossProductView.hpp"
#include "views/reductions/CoefficientSum.hpp"
#include "views/unary/HomogeneousView.hpp"

namespace zipper {
template <typename ValueType, index_type Rows>
class Vector;

template <concepts::ViewDerived View>
class VectorBase : public ZipperBase<VectorBase, View> {
   public:
    VectorBase() = default;

    using view_type = View;
    using value_type = View::value_type;
    using extents_type = View::extents_type;
    using traits = zipper::views::detail::ViewTraits<view_type>;
    using extents_traits = detail::ExtentsTraits<extents_type>;
    static_assert(extents_traits::rank == 1);
    using Base = ZipperBase<VectorBase, View>;

    using Base::Base;
    // using Base::operator=;
    using Base::cast;
    using Base::swizzle;
    using Base::view;

    auto eval() const { return Vector(*this); }
    VectorBase& operator=(concepts::ViewDerived auto const& v) {
        return Base::operator=(v);
    }
    VectorBase& operator=(concepts::VectorBaseDerived auto const& v) {
        return operator=(v.view());
    }

    // TODO: make vectorbase or zipperbase assignable from initializer lists
    // template <typename T>
    // VectorBase& operator=(const std::initializer_list<T>& l)
    //    requires(extents_traits::is_dynamic)
    //{
    //    view().resize(extents_type(l.size()));
    //    std::ranges::copy(l, begin());
    //}

    template <concepts::VectorBaseDerived Other>
    VectorBase(const Other& other)
        requires(view_type::is_writable)
        : VectorBase(other.view()) {}

    template <concepts::VectorBaseDerived Other>
    VectorBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        return operator=(other.view());
    }

    // auto as_col_matrix() const {
    //     return MatrixBase<views::unary::IdentityView<View>>(view());
    // }
    auto as_array() const { return zipper::as_array(*this); }
    auto as_tensor() const { return zipper::as_tensor(*this); }
    auto as_form() const { return zipper::as_form(*this); }

    template <index_type T>
    value_type norm_powered() const {
        if constexpr (T == 1) {
            return views::reductions::CoefficientSum{as_array().abs().view()}();
        } else if constexpr (T == 2) {
            auto arr = as_array();
            return views::reductions::CoefficientSum{(arr * arr).view()}();
        } else {
            return views::reductions::CoefficientSum{
                as_array().pow(T).abs().view()}();
        }
    }
    value_type norm_powered(value_type T) const {
        return views::reductions::CoefficientSum{
            as_array().pow(T).abs().view()}();
    }

    value_type dot(concepts::VectorBaseDerived auto const& o) const {
        return as_form() * o;
    }
    template <concepts::VectorBaseDerived O>
    auto cross(O const& o) const {
        return VectorBase<
            views::binary::CrossProductView<View, typename O::view_type>>(
            views::binary::CrossProductView<View, typename O::view_type>(
                view(), o.view()));
    }

    template <index_type I>
    auto head()
        requires(view_type::is_writable)
    {
        auto S = slice(std::integral_constant<index_type, 0>{},
                       std::integral_constant<index_type, I>{});
        auto v = Base::slice_view(S);
        using V = std::decay_t<decltype(v)>;
        return VectorBase<V>(std::move(v));
    }
    template <index_type I>
    auto head() const {
        auto S = slice(std::integral_constant<index_type, 0>{},
                       std::integral_constant<index_type, I>{});
        auto v = Base::slice_view(S);
        using V = std::decay_t<decltype(v)>;
        return VectorBase<V>(std::move(v));
    }
    auto head(index_type N)
        requires(view_type::is_writable)
    {
        auto S = slice<std::integral_constant<index_type, 0>, index_type>(
            std::integral_constant<index_type, 0>{}, N);
        auto v = Base::slice_view(S);
        using V = std::decay_t<decltype(v)>;
        return VectorBase<V>(std::move(v));
    }
    auto head(index_type N) const {
        auto S = slice(std::integral_constant<index_type, 0>{}, N);
        auto v = Base::slice_view(S);
        using V = std::decay_t<decltype(v)>;
        return VectorBase<V>(std::move(v));
    }
    // TODO: arithmetic
    // template <index_type I>
    // auto tail()
    //    requires(view_type::is_writable)
    //{
    //    auto S = slice(std::integral_constant<index_type, 0>{},
    //                   std::integral_constant<index_type, I>{});
    //    auto v = Base::slice_view(S);
    //    using V = std::decay_t<decltype(v)>;
    //    return VectorBase<V>(std::move(v));
    //}
    // template <index_type I>
    // auto tail() const {
    //    auto S = slice(std::integral_constant<index_type, 0>{},
    //                   std::integral_constant<index_type, I>{});
    //    auto v = Base::slice_view(S);
    //    using V = std::decay_t<decltype(v)>;
    //    return VectorBase<V>(std::move(v));
    //}
    // auto tail(index_type N)
    //    requires(view_type::is_writable)
    //{
    //    auto S = slice<std::integral_constant<index_type, 0>, index_type>(
    //        std::integral_constant<index_type, 0>{}, N);
    //    auto v = Base::slice_view(S);
    //    using V = std::decay_t<decltype(v)>;
    //    return VectorBase<V>(std::move(v));
    //}
    // auto tail(index_type N) const {
    //    auto S = slice(std::integral_constant<index_type, 0>{}, N);
    //    auto v = Base::slice_view(S);
    //    using V = std::decay_t<decltype(v)>;
    //    return VectorBase<V>(std::move(v));
    //}

    template <index_type T = 2>
    value_type norm() const {
        const value_type v = norm_powered<T>();
        if constexpr (T == 1) {
            return v;
        } else if constexpr (T == 2) {
            return std::sqrt(v);
        } else {
            const value_type p = value_type(1.0) / T;
            return std::pow<value_type>(v, p);
        }
    }
    value_type norm(value_type T) const {
        value_type p = value_type(1.0) / T;
        return std::pow<value_type>(norm_powered(T), p);
    }

    template <index_type T = 2>
    auto normalized() const {
        return *this / norm<T>();
    }
    value_type normalized(value_type T) const { return *this / norm(T); }
    template <index_type T = 2>
    void normalize() {
        *this /= norm<T>();
    }
    void normalize(value_type T) { *this /= norm(T); }

    template <views::unary::HomogeneousMode Mode =
                  views::unary::HomogeneousMode::Position>
    auto homogeneous() const {
        return VectorBase<views::unary::HomogeneousView<Mode, View>>(view());
    }
};

template <concepts::VectorViewDerived View>
VectorBase(View&& view) -> VectorBase<View>;
template <concepts::VectorViewDerived View>
VectorBase(const View& view) -> VectorBase<View>;

UNARY_DECLARATION(VectorBase, LogicalNot, operator!)
UNARY_DECLARATION(VectorBase, BitNot, operator~)
UNARY_DECLARATION(VectorBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(VectorBase, Divides, operator/)

BINARY_DECLARATION(VectorBase, Plus, operator+)
BINARY_DECLARATION(VectorBase, Minus, operator-)

template <concepts::VectorBaseDerived View1, concepts::VectorBaseDerived View2>
bool operator==(View1 const& lhs, View2 const& rhs) {
    return (lhs.as_array() == rhs.as_array()).all();
}

template <concepts::VectorBaseDerived View1, concepts::VectorBaseDerived View2>
bool operator!=(View1 const& lhs, View2 const& rhs) {
    return (lhs.as_array() != rhs.as_array()).any();
}
template <concepts::VectorBaseDerived View>
auto operator*(View const& lhs, typename View::value_type const& rhs) {
    using V =
        views::unary::ScalarMultipliesView<typename View::value_type,
                                           typename View::view_type, true>;
    return VectorBase<V>(V(lhs.view(), rhs));
}
template <concepts::VectorBaseDerived View>
auto operator*(typename View::value_type const& lhs, View const& rhs) {
    using V =
        views::unary::ScalarMultipliesView<typename View::value_type,
                                           typename View::view_type, false>;
    return VectorBase<V>(V(lhs, rhs.view()));
}

}  // namespace zipper
#endif
