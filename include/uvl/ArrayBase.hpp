#if !defined(UVL_ARRAYBASE_HPP)
#define UVL_ARRAYBASE_HPP

#include "uvl/detail/convert_extents.hpp"
#include "uvl/types.hpp"
//
#include "concepts/ArrayBaseDerived.hpp"
#include "concepts/ViewDerived.hpp"
//
#include "views/binary/AdditionView.hpp"
#include "views/binary/CoeffProductView.hpp"
#include "views/unary/AbsoluteView.hpp"
#include "views/unary/CastView.hpp"
#include "views/unary/ScalarPowerView.hpp"
#include "views/unary/ScalarProductView.hpp"
#include "views/unary/SwizzleView.hpp"

namespace uvl {

template <concepts::ViewDerived View>
class ArrayBase {
   public:
    ArrayBase() = default;

    using view_type = View;
    using value_type = View::value_type;
    using extents_type = View::extents_type;
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

    template <concepts::ArrayBaseDerived Other>
    ArrayBase(const Other& other)
        requires(view_type::is_writable)
        : m_view(detail::convert_extents<extents_type>(other.extents())) {
        m_view.assign(other.view());
    }
    template <concepts::ArrayBaseDerived Other>
    ArrayBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        m_view.assign(other.view());
    }

    template <concepts::ArrayBaseDerived Other>
    friend auto operator+(const ArrayBase<view_type>& lhs, Other const& rhs) {
        return ArrayBase<
            views::binary::AdditionView<view_type, typename Other::view_type>>(
            lhs.view(), rhs.view());
    }
    template <concepts::ArrayBaseDerived Other>
    friend auto operator*(const ArrayBase<view_type>& lhs, Other const& rhs) {
        return ArrayBase<views::binary::CoeffProductView<
            view_type, typename Other::view_type>>(lhs.view(), rhs.view());
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

    auto pow(value_type const& exp) const {
        return ArrayBase<views::unary::ScalarPowerView<view_type, value_type>>(
            view(), exp);
    }

    auto abs() const {
        return ArrayBase<views::unary::AbsoluteView<view_type>>(view());
    }

    template <typename T>
    auto cast() const {
        return ArrayBase<views::unary::CastView<T, view_type>>(view());
    }

    auto transpose() const {
        return ArrayBase<views::unary::SwizzleView<view_type, 1, 0>>(view());
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

template <concepts::ViewDerived View>
ArrayBase(View&& view) -> ArrayBase<View>;
template <concepts::ViewDerived View>
ArrayBase(const View& view) -> ArrayBase<View>;

}  // namespace uvl

#endif
