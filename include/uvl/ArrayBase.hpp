#if !defined(UVL_ARRAYBASE_HPP)
#define UVL_ARRAYBASE_HPP

#include "uvl/detail/convert_extents.hpp"
#include "uvl/types.hpp"
#include "views/reductions/CoefficientSum.hpp"
//
#include "concepts/ArrayBaseDerived.hpp"
#include "concepts/ViewDerived.hpp"
//
#include "views/binary/AdditionView.hpp"
#include "views/unary/CastView.hpp"
#include "views/unary/NegateView.hpp"
#include "views/unary/ScalarProductView.hpp"
#include "views/unary/ScalarQuotientView.hpp"
#include "views/unary/SwizzleView.hpp"
////
#include "views/binary/CoeffProductView.hpp"
#include "views/unary/AbsoluteView.hpp"
#include "views/unary/ScalarPowerView.hpp"

namespace uvl {

template <concepts::ViewDerived View>
class ArrayBase {
   public:
    ArrayBase() = default;

    using view_type = View;
    using value_type = View::value_type;
    using extents_type = View::extents_type;

    template <index_type... N>
    auto eval(const extents<N...>&) const {
        return Array<value_type, N...>(*this);
    }
    auto eval() const { return eval(extents()); }
    template <typename... Args>
    ArrayBase(Args&&... v) : m_view(std::forward<Args>(v)...) {}

    ArrayBase(View&& v) : m_view(v) {}
    ArrayBase(const View& v) : m_view(v) {}
    ArrayBase& operator=(View&& v) { m_view = v; }
    ArrayBase& operator=(const View& v) { m_view = v; }

    ArrayBase(ArrayBase&& v) = default;
    ArrayBase(const ArrayBase& v) = default;
    ArrayBase& operator=(ArrayBase&& v) = default;
    ArrayBase& operator=(const ArrayBase& v) = default;

    template <concepts::ViewDerived Other>
    ArrayBase(const Other& other)
        requires(view_type::is_writable)
        : m_view(detail::convert_extents<extents_type>(other.extents())) {
        m_view.assign(other);
    }
    template <concepts::ViewDerived Other>
    ArrayBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        m_view.assign(other);
        return *this;
    }
    template <concepts::ArrayBaseDerived Other>
    ArrayBase(const Other& other)
        requires(view_type::is_writable)
        : ArrayBase(other.view()) {}
    template <concepts::ArrayBaseDerived Other>
    ArrayBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        return operator=(other.view());
    }

    ArrayBase& operator*=(const value_type& other)
        requires(view_type::is_writable)
    {
        return *this = other * *this;
    }
    ArrayBase& operator/=(const value_type& other)
        requires(view_type::is_writable)
    {
        return *this = *this / other;
    }
    template <concepts::ArrayBaseDerived Other>
    ArrayBase& operator+=(const Other& other)
        requires(view_type::is_writable)
    {
        return *this = *this + other;
    }
    template <concepts::ArrayBaseDerived Other>
    ArrayBase& operator-=(const Other& other)
        requires(view_type::is_writable)
    {
        return *this = *this - other;
    }

    // SCALAR STUFF
    template <concepts::ArrayBaseDerived Other>
    friend auto operator+(const ArrayBase<view_type>& lhs, Other const& rhs) {
        return ArrayBase<
            views::binary::AdditionView<view_type, typename Other::view_type>>(
            lhs.view(), rhs.view());
    }
    friend auto operator-(const ArrayBase<view_type>& lhs) {
        return ArrayBase<views::unary::NegateView<view_type>>(lhs.view());
    }
    template <concepts::ArrayBaseDerived Other>
    friend auto operator-(const ArrayBase<view_type>& lhs, Other const& rhs) {
        return lhs + (-rhs);
    }

    friend auto operator*(const ArrayBase<view_type>& lhs,
                          value_type const& rhs) {
        return ArrayBase<
            views::unary::ScalarProductView<value_type, view_type>>(rhs,
                                                                    lhs.view());
    }
    friend auto operator*(value_type const& lhs,
                          const ArrayBase<view_type>& rhs) {
        return ArrayBase<
            views::unary::ScalarProductView<value_type, view_type>>(lhs,
                                                                    rhs.view());
    }
    friend auto operator/(const ArrayBase<view_type>& lhs,
                          value_type const& rhs) {
        return ArrayBase<
            views::unary::ScalarQuotientView<view_type, value_type>>(lhs.view(),
                                                                     rhs);
    }

    template <rank_type... ranks>
    auto swizzle() const {
        return ArrayBase<views::unary::SwizzleView<value_type, ranks...>>(
            view());
    }
    template <typename T>
    auto cast() const {
        return ArrayBase<views::unary::CastView<T, view_type>>(view());
    }

    //--------------------------------------------------------------
    //
    template <concepts::ArrayBaseDerived Other>
    friend auto operator*(const ArrayBase<view_type>& lhs, Other const& rhs) {
        return ArrayBase<views::binary::CoeffProductView<
            view_type, typename Other::view_type>>(lhs.view(), rhs.view());
    }

    auto pow(value_type const& exp) const {
        return ArrayBase<views::unary::ScalarPowerView<view_type, value_type>>(
            view(), exp);
    }

    auto abs() const {
        return ArrayBase<views::unary::AbsoluteView<view_type>>(view());
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
        if constexpr(T == 1) {
            return v;
        } else if constexpr(T == 2) {
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

template <concepts::ViewDerived View>
ArrayBase(View&& view) -> ArrayBase<View>;
template <concepts::ViewDerived View>
ArrayBase(const View& view) -> ArrayBase<View>;

}  // namespace uvl

#endif
