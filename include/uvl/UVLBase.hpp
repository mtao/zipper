#if !defined(UVL_UVLBASE_HPP)
#define UVL_UVLBASE_HPP

#include "concepts/UVLBaseDerived.hpp"
#include "concepts/ViewDerived.hpp"
#include "uvl/detail/declare_operations.hpp"
#include "uvl/types.hpp"
#include "uvl/views/binary/ArithmeticViews.hpp"
#include "uvl/views/unary/ScalarArithmeticViews.hpp"
#include "views/unary/CastView.hpp"
#include "views/unary/DiagonalView.hpp"
#include "views/unary/SliceView.hpp"
#include "views/unary/SwizzleView.hpp"
#include "views/unary/detail/operation_implementations.hpp"

namespace uvl {

template <template <concepts::ViewDerived> typename DerivedT,
          concepts::ViewDerived View>
class UVLBase {
   public:
    UVLBase() = default;
    using Derived = DerivedT<View>;
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
    Derived& derived() { return static_cast<Derived&>(*this); }

    using view_type = View;
    using view_traits = views::detail::ViewTraits<view_type>;
    using value_type = View::value_type;
    using extents_type = View::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;

    const View& view() const { return m_view; }
    View& view() { return m_view; }
    const extents_type& extents() const { return view().extents(); }
    constexpr index_type extent(rank_type i) const { return m_view.extent(i); }
    template <typename... Args>
    UVLBase(Args&&... v) : m_view(std::forward<Args>(v)...) {}

    UVLBase(View&& v) : m_view(v) {}
    UVLBase(const View& v) : m_view(v) {}
    Derived& operator=(concepts::ViewDerived auto const& v) {
        m_view = v;
        return derived();
    }

    UVLBase(UVLBase&& v) = default;
    UVLBase(const UVLBase& v) = default;

    template <concepts::ViewDerived Other>
    UVLBase(const Other& other)
        requires(view_type::is_writable &&
                 detail::extents::assignable_extents_v<
                     typename Other::extents_type, extents_type>)
        : m_view(extents_traits::convert_from(other.extents())) {
        m_view.assign(other);
    }
    template <concepts::ViewDerived Other>
    Derived& operator=(const Other& other)
        requires(view_type::is_writable &&
                 detail::extents::assignable_extents_v<
                     typename Other::extents_type, extents_type>)
    {
        m_view.assign(other);
        return derived();
    }

    template <concepts::UVLBaseDerived Other>
    Derived& operator+=(const Other& other)
        requires(view_type::is_writable)
    {
        *this = *this + other;
    }
    template <concepts::UVLBaseDerived Other>
    Derived& operator-=(const Other& other)
        requires(view_type::is_writable)
    {
        return *this = *this - other;
    }
    Derived& operator*=(const value_type& other)
        requires(view_type::is_writable)
    {
        return *this = other * *this;
    }
    Derived& operator/=(const value_type& other)
        requires(view_type::is_writable)
    {
        return *this = *this / other;
    }

    template <template <typename> typename BaseType = DerivedT,
              rank_type... ranks>
    auto swizzle() const {
        using V = views::unary::SwizzleView<view_type, ranks...>;
        return BaseType<V>(V(view()));
    }
    template <typename T>
    auto cast() const {
        using V = views::unary::CastView<T, view_type>;
        return DerivedT<V>(V(view()));
    }

    template <typename... Args>
    auto operator()(Args&&... idxs) -> decltype(auto)
        requires(view_traits::is_writable)

    {
        return view()(std::forward<Args>(idxs)...);
    }
    template <typename... Args>
    auto operator()(Args&&... idxs) const -> decltype(auto)

    {
        return view()(std::forward<Args>(idxs)...);
    }

   protected:
    // slicing has fairly dimension specific effects for most derived types,
    // so we will just return the view and let base class return things
    template <typename... Slices>
    auto slice_view(Slices&&... slices) const {
        using view_type = views::unary::SliceView<view_type, true, Slices...>;

        return view_type(view(), std::forward<Slices>(slices)...);
    }
    template <typename... Slices>
    auto slice_view() const {
        using view_type = views::unary::SliceView<view_type, true, Slices...>;
        return view_type(view(), Slices{}...);
    }

    template <typename... Slices>
    auto slice_view(Slices&&... slices) {
        using view_type = views::unary::SliceView<view_type, false, Slices...>;
        return view_type(view(), std::forward<Slices>(slices)...);
    }
    template <typename... Slices>
    auto slice_view() {
        using view_type = views::unary::SliceView<view_type, false, Slices...>;
        return view_type(view(), Slices{}...);
    }

   public:
   private:
    View m_view;
};

UNARY_DECLARATION(UVLBase, LogicalNot, operator!)
UNARY_DECLARATION(UVLBase, BitNot, operator~)
UNARY_DECLARATION(UVLBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(UVLBase, Plus, operator+)
SCALAR_BINARY_DECLARATION(UVLBase, Minus, operator-)
SCALAR_BINARY_DECLARATION(UVLBase, Multiplies, operator*)
SCALAR_BINARY_DECLARATION(UVLBase, Divides, operator/)
// SCALAR_BINARY_DECLARATION(UVLBase, Modulus, operator%)
// SCALAR_BINARY_DECLARATION(UVLBase, EqualsTo, operator==)
// SCALAR_BINARY_DECLARATION(UVLBase, NotEqualsTo, operator!=)
// SCALAR_BINARY_DECLARATION(UVLBase, Greater, operator>)
// SCALAR_BINARY_DECLARATION(UVLBase, Less, operator<)
// SCALAR_BINARY_DECLARATION(UVLBase, GreaterEqual, operator>=)
// SCALAR_BINARY_DECLARATION(UVLBase, LessEqual, operator<=)
// SCALAR_BINARY_DECLARATION(UVLBase, LogicalAnd, operator&&)
// SCALAR_BINARY_DECLARATION(UVLBase, LogicalOr, operator||)
// SCALAR_BINARY_DECLARATION(UVLBase, BitAnd, operator&)
// SCALAR_BINARY_DECLARATION(UVLBase, BitOr, operator|)
// SCALAR_BINARY_DECLARATION(UVLBase, BitXor, operator^)

BINARY_DECLARATION(UVLBase, Plus, operator+)
BINARY_DECLARATION(UVLBase, Minus, operator-)
// BINARY_DECLARATION(UVLBase, Divides, operator/)
// BINARY_DECLARATION(UVLBase, Modulus, operator%)
// BINARY_DECLARATION(UVLBase, EqualsTo, operator==)
// BINARY_DECLARATION(UVLBase, NotEqualsTo, operator!=)
// BINARY_DECLARATION(UVLBase, Greater, operator>)
// BINARY_DECLARATION(UVLBase, Less, operator<)
// BINARY_DECLARATION(UVLBase, GreaterEqual, operator>=)
// BINARY_DECLARATION(UVLBase, LessEqual, operator<=)
// BINARY_DECLARATION(UVLBase, LogicalAnd, operator&&)
// BINARY_DECLARATION(UVLBase, LogicalOr, operator||)
// BINARY_DECLARATION(UVLBase, BitAnd, operator&)
// BINARY_DECLARATION(UVLBase, BitOr, operator|)
// BINARY_DECLARATION(UVLBase, BitXor, operator^)

template <concepts::UVLBaseDerived View1, concepts::UVLBaseDerived View2>
bool operator==(View1 const& lhs, View2 const& rhs) {
    return (lhs.as_array() == rhs.as_array()).all();
}

template <concepts::UVLBaseDerived View1, concepts::UVLBaseDerived View2>
bool operator!=(View1 const& lhs, View2 const& rhs) {
    return (lhs.as_array() != rhs.as_array()).any();
}
}  // namespace uvl

#endif
