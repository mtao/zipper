#if !defined(UVL_FORMBASE_HPP)
#define UVL_FORMBASE_HPP

#include "concepts/FormBaseDerived.hpp"
#include "concepts/ViewDerived.hpp"
#include "uvl/detail/declare_operations.hpp"
#include "uvl/types.hpp"
#include "uvl/views/binary/ArithmeticViews.hpp"
#include "uvl/views/binary/WedgeProductView.hpp"
#include "uvl/views/unary/ScalarArithmeticViews.hpp"
#include "views/reductions/CoefficientSum.hpp"
#include "views/unary/AbsView.hpp"
#include "views/unary/CastView.hpp"
#include "views/unary/DiagonalView.hpp"
#include "views/unary/ScalarPowerView.hpp"
#include "views/unary/SliceView.hpp"
#include "views/unary/SwizzleView.hpp"
#include "views/unary/detail/operation_implementations.hpp"

namespace uvl {

template <concepts::ViewDerived View>
class FormBase {
   public:
    FormBase() = default;

    using view_type = View;
    using value_type = View::value_type;
    using extents_type = View::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;

    template <index_type... N>
    auto eval(const extents<N...>&) const {
        return Form<value_type, N...>(*this);
    }
    auto eval() const { return eval(extents()); }
    template <typename... Args>
    FormBase(Args&&... v) : m_view(std::forward<Args>(v)...) {}

    FormBase(View&& v) : m_view(v) {}
    FormBase(const View& v) : m_view(v) {}
    FormBase& operator=(concepts::ViewDerived auto const& v) {
        m_view = v;
        return *this;
    }
    FormBase& operator=(concepts::FormBaseDerived auto const& v) {
        m_view = v.view();
        return *this;
    }

    FormBase(FormBase&& v) = default;
    FormBase(const FormBase& v) = default;

    template <concepts::ViewDerived Other>
    FormBase(const Other& other)
        requires(view_type::is_writable)
        : m_view(extents_traits::convert_from(other.extents())) {
        m_view.assign(other);
    }
    template <concepts::ViewDerived Other>
    FormBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        m_view.assign(other);
        return *this;
    }
    template <concepts::FormBaseDerived Other>
    FormBase(const Other& other)
        requires(view_type::is_writable)
        : FormBase(other.view()) {}
    template <concepts::FormBaseDerived Other>
    FormBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        return operator=(other.view());
    }

    FormBase& operator*=(const value_type& other)
        requires(view_type::is_writable)
    {
        return *this = other * *this;
    }
    FormBase& operator/=(const value_type& other)
        requires(view_type::is_writable)
    {
        return *this = *this / other;
    }
    template <concepts::FormBaseDerived Other>
    FormBase& operator+=(const Other& other)
        requires(view_type::is_writable)
    {
        return *this = *this + other;
    }
    template <concepts::FormBaseDerived Other>
    FormBase& operator-=(const Other& other)
        requires(view_type::is_writable)
    {
        return *this = *this - other;
    }

    template <rank_type... ranks>
    auto swizzle() const {
        return FormBase<views::unary::SwizzleView<value_type, ranks...>>(
            view());
    }
    template <typename T>
    auto cast() const {
        return FormBase<views::unary::CastView<T, view_type>>(view());
    }

    //--------------------------------------------------------------
    //

    auto pow(value_type const& exp) const {
        return FormBase<views::unary::ScalarPowerView<view_type, value_type>>(
            view(), exp);
    }

    auto abs() const {
        return FormBase<views::unary::AbsView<view_type>>(view());
    }

    template <index_type T>
    value_type norm_powered() const {
        if constexpr (T == 1) {
            return views::reductions::CoefficientSum{abs().view()}();
        } else if constexpr (T == 2) {
            return views::reductions::CoefficientSum{(*this * *this).view()}();
        } else {
            return views::reductions::CoefficientSum{pow(T).abs().view()}();
        }
    }
    value_type norm_powered(value_type T) const {
        return views::reductions::CoefficientSum{pow(T).abs().view()}();
    }

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

    template <typename... Args>
    value_type operator()(Args&&... idxs) const

    {
        return view()(std::forward<Args>(idxs)...);
    }

    template <typename... Slices>
    auto slice(Slices&&... slices) const {
        using view_type = views::unary::SliceView<view_type, true, Slices...>;

        return FormBase<view_type>(
            view_type(view(), std::forward<Slices>(slices)...));
    }
    template <typename... Slices>
    auto slice() const {
        using view_type = views::unary::SliceView<view_type, true, Slices...>;
        return FormBase<view_type>(view_type(view(), Slices{}...));
    }

    template <typename... Slices>
    auto slice(Slices&&... slices) {
        using view_type = views::unary::SliceView<view_type, false, Slices...>;
        return FormBase<view_type>(
            view_type(view(), std::forward<Slices>(slices)...));
    }
    template <typename... Slices>
    auto slice() {
        using view_type = views::unary::SliceView<view_type, false, Slices...>;
        return FormBase<view_type>(view_type(view(), Slices{}...));
    }

    auto diagonal() const {
        return VectorBase<views::unary::DiagonalView<view_type, true>>(view());
    }
    auto diagonal() {
        return VectorBase<views::unary::DiagonalView<view_type, false>>(view());
    }

    const View& view() const { return m_view; }
    View& view() { return m_view; }
    const extents_type& extents() const { return view().extents(); }
    constexpr index_type extent(rank_type i) const { return m_view.extent(i); }

   private:
    View m_view;
};

template <concepts::ViewDerived View>
FormBase(View&& view) -> FormBase<View>;
template <concepts::ViewDerived View>
FormBase(const View& view) -> FormBase<View>;

UNARY_DECLARATION(FormBase, LogicalNot, operator!)
UNARY_DECLARATION(FormBase, BitNot, operator~)
UNARY_DECLARATION(FormBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(FormBase, Plus, operator+)
SCALAR_BINARY_DECLARATION(FormBase, Minus, operator-)
SCALAR_BINARY_DECLARATION(FormBase, Multiplies, operator*)
SCALAR_BINARY_DECLARATION(FormBase, Divides, operator/)
// SCALAR_BINARY_DECLARATION(FormBase, Modulus, operator%)
// SCALAR_BINARY_DECLARATION(FormBase, EqualsTo, operator==)
// SCALAR_BINARY_DECLARATION(FormBase, NotEqualsTo, operator!=)
// SCALAR_BINARY_DECLARATION(FormBase, Greater, operator>)
// SCALAR_BINARY_DECLARATION(FormBase, Less, operator<)
// SCALAR_BINARY_DECLARATION(FormBase, GreaterEqual, operator>=)
// SCALAR_BINARY_DECLARATION(FormBase, LessEqual, operator<=)
// SCALAR_BINARY_DECLARATION(FormBase, LogicalAnd, operator&&)
// SCALAR_BINARY_DECLARATION(FormBase, LogicalOr, operator||)
// SCALAR_BINARY_DECLARATION(FormBase, BitAnd, operator&)
// SCALAR_BINARY_DECLARATION(FormBase, BitOr, operator|)
// SCALAR_BINARY_DECLARATION(FormBase, BitXor, operator^)

BINARY_DECLARATION(FormBase, Plus, operator+)
BINARY_DECLARATION(FormBase, Minus, operator-)
// BINARY_DECLARATION(FormBase, Divides, operator/)
// BINARY_DECLARATION(FormBase, Modulus, operator%)
// BINARY_DECLARATION(FormBase, EqualsTo, operator==)
// BINARY_DECLARATION(FormBase, NotEqualsTo, operator!=)
// BINARY_DECLARATION(FormBase, Greater, operator>)
// BINARY_DECLARATION(FormBase, Less, operator<)
// BINARY_DECLARATION(FormBase, GreaterEqual, operator>=)
// BINARY_DECLARATION(FormBase, LessEqual, operator<=)
// BINARY_DECLARATION(FormBase, LogicalAnd, operator&&)
// BINARY_DECLARATION(FormBase, LogicalOr, operator||)
// BINARY_DECLARATION(FormBase, BitAnd, operator&)
// BINARY_DECLARATION(FormBase, BitOr, operator|)
// BINARY_DECLARATION(FormBase, BitXor, operator^)

template <concepts::FormBaseDerived View1, concepts::FormBaseDerived View2>
bool operator==(View1 const& lhs, View2 const& rhs) {
    return (lhs.as_array() == rhs.as_array()).all();
}

template <concepts::FormBaseDerived View1, concepts::FormBaseDerived View2>
bool operator!=(View1 const& lhs, View2 const& rhs) {
    return (lhs.as_array() != rhs.as_array()).any();
}

template <concepts::FormBaseDerived View1, concepts::FormBaseDerived View2>
auto operator*(View1 const& lhs, View2 const& rhs) {
    return FormBase<views::binary::WedgeProductView<
        typename View1::view_type, typename View2::view_type>>(lhs.view(),
                                                               rhs.view());
}
}  // namespace uvl

#endif
