
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
#include "views/binary/AdditionView.hpp"
#include "views/unary/CastView.hpp"
#include "views/unary/NegateView.hpp"
#include "views/unary/IdempotentView.hpp"
#include "views/unary/ScalarProductView.hpp"
#include "views/unary/ScalarQuotientView.hpp"
#include "views/unary/SwizzleView.hpp"
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

    auto eval() const {
        return Vector(*this);
    }

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


    template <concepts::VectorBaseDerived Other>
    friend auto operator+(const VectorBase<view_type>& lhs, Other const& rhs) {
        return VectorBase<
            views::binary::AdditionView<view_type, typename Other::view_type>>(
            lhs.view(), rhs.view());
    }
    friend auto operator-(const VectorBase<view_type>& lhs) {
        return VectorBase<views::unary::NegateView<view_type>>(lhs.view());
    }
    template <concepts::VectorBaseDerived Other>
    friend auto operator-(const VectorBase<view_type>& lhs, Other const& rhs) {
        return lhs + (-rhs);
    }

    friend auto operator*(const VectorBase<view_type>& lhs,
                          value_type const& rhs) {
        return VectorBase<
            views::unary::ScalarProductView<value_type, view_type>>(rhs,
                                                                    lhs.view());
    }
    friend auto operator*(value_type const& lhs,
                          const VectorBase<view_type>& rhs) {
        return VectorBase<
            views::unary::ScalarProductView<value_type, view_type>>(lhs,
                                                                    rhs.view());
    }
    friend auto operator/(const VectorBase<view_type>& lhs,
                          value_type const& rhs) {
        return VectorBase<
            views::unary::ScalarQuotientView<view_type, value_type>>(lhs.view(),
                                                                     rhs);
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

    auto as_array() const { return ArrayBase<views::unary::IdempotentView<View>>(view()); }
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
    value_type normalized(value_type T) const {
        return *this / norm(T);
    }
    template <index_type T = 2>
    void normalize() {
        *this /= norm<T>();

    }
    void normalize(value_type T) {
        *this /= norm(T);
    }

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

}  // namespace uvl
#endif
