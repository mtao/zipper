
#if !defined(UVL_VECTORBASE_HPP)
#define UVL_VECTORBASE_HPP

#include <cmath>

#include "uvl/detail/convert_extents.hpp"
#include "uvl/types.hpp"
//
#include "concepts/VectorBaseDerived.hpp"
#include "concepts/VectorViewDerived.hpp"
//
// Scalar stuff
#include "uvl/detail/declare_operations.hpp"
#include "uvl/views/binary/ArithmeticViews.hpp"
#include "views/unary/CastView.hpp"
#include "views/unary/IdentityView.hpp"
#include "views/unary/ScalarArithmeticViews.hpp"
#include "views/unary/detail/operation_implementations.hpp"
//
#include "ArrayBase.hpp"
#include "views/reductions/CoefficientSum.hpp"

namespace uvl {
template <typename ValueType, index_type Rows>
class Vector;

template <concepts::VectorViewDerived View>
class VectorBase {
   public:
    VectorBase() = default;

    using view_type = View;
    using value_type = View::value_type;
    using extents_type = View::extents_type;
    template <typename... Args>
    VectorBase(Args&&... v) : m_view(std::forward<Args>(v)...) {}

    VectorBase(View&& v) : m_view(v) {}
    VectorBase(const View& v) : m_view(v) {}
    VectorBase& operator=(View&& v) { m_view = v; }
    VectorBase& operator=(const View& v) { m_view = v; }

    VectorBase(VectorBase&& v) = default;
    VectorBase(const VectorBase& v) = default;
    VectorBase& operator=(VectorBase&& v) = default;
    VectorBase& operator=(const VectorBase& v) = default;

    auto eval() const { return Vector(*this); }

    template <concepts::VectorViewDerived Other>
    VectorBase(const Other& other)
        requires(view_type::is_writable)
        : m_view(detail::convert_extents<extents_type>(other.extents())) {
        m_view.assign(other);
    }
    template <concepts::VectorViewDerived Other>
    VectorBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        m_view.assign(other);
        return *this;
    }
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
    VectorBase& operator*=(const value_type& other)
        requires(view_type::is_writable)
    {
        return *this = other * *this;
    }
    VectorBase& operator/=(const value_type& other)
        requires(view_type::is_writable)
    {
        return *this = *this / other;
    }
    template <concepts::VectorBaseDerived Other>
    VectorBase& operator+=(const Other& other)
        requires(view_type::is_writable)
    {
        return *this = *this + other;
    }
    template <concepts::VectorBaseDerived Other>
    VectorBase& operator-=(const Other& other)
        requires(view_type::is_writable)
    {
        return *this = *this - other;
    }

    // template <rank_type... ranks>
    // auto swizzle() const {
    //     return VectorBase<views::unary::SwizzleView<value_type, ranks...>>(
    //         view());
    // }
    template <typename T>
    auto cast() const {
        return VectorBase<views::unary::CastView<T, view_type>>(view());
    }

    auto as_array() const {
        return ArrayBase<views::unary::IdentityView<View>>(view());
    }
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

    const View& view() const { return m_view; }
    View& view() { return m_view; }
    const extents_type& extents() const { return view().extents(); }
    constexpr index_type extent(rank_type i) const { return m_view.extent(i); }

   private:
    View m_view;
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
BINARY_DECLARATION(VectorBase, EqualsTo, operator==)
BINARY_DECLARATION(VectorBase, NotEqualsTo, operator!=)
BINARY_DECLARATION(VectorBase, Greater, operator>)
BINARY_DECLARATION(VectorBase, Less, operator<)
BINARY_DECLARATION(VectorBase, GreaterEqual, operator>=)
BINARY_DECLARATION(VectorBase, LessEqual, operator<=)
BINARY_DECLARATION(VectorBase, LogicalAnd, operator&&)
BINARY_DECLARATION(VectorBase, LogicalOr, operator||)
BINARY_DECLARATION(VectorBase, BitAnd, operator&)
BINARY_DECLARATION(VectorBase, BitOr, operator|)
BINARY_DECLARATION(VectorBase, BitXor, operator^)
template <concepts::VectorBaseDerived View>
auto operator*(View const& lhs, typename View::value_type const& rhs) {
    return VectorBase<views::unary::ScalarMultipliesView<
        typename View::value_type, typename View::view_type, true>>(lhs.view(),
                                                                    rhs);
}
template <concepts::VectorBaseDerived View>
auto operator*(typename View::value_type const& lhs, View const& rhs) {
    return VectorBase<views::unary::ScalarMultipliesView<
        typename View::value_type, typename View::view_type, false>>(
        lhs, rhs.view());
}
}  // namespace uvl
#endif
