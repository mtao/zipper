#if !defined(ZIPPER_ZIPPERBASE_HPP)
#define ZIPPER_ZIPPERBASE_HPP

#include "concepts/ViewDerived.hpp"
#include "concepts/ZipperBaseDerived.hpp"
#include "views/unary/CastView.hpp"
#include "views/unary/DiagonalView.hpp"
#include "views/unary/SliceView.hpp"
#include "views/unary/SwizzleView.hpp"
#include "views/unary/detail/operation_implementations.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/types.hpp"
#include "zipper/views/binary/ArithmeticViews.hpp"
#include "zipper/views/unary/ScalarArithmeticViews.hpp"

namespace zipper {

template <template <concepts::ViewDerived> typename DerivedT,
          concepts::ViewDerived View>
class ZipperBase {
   public:
    ZipperBase() = default;
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
    static constexpr index_type static_extent(rank_type i) {
        return View::static_extent(i);
    }
    // template <typename... Args>
    // ZipperBase(Args&&... v) : m_view(std::forward<Args>(v)...) {}

    ZipperBase(View&& v) : m_view(std::move(v)) {}
    ZipperBase(Derived&& v) : ZipperBase(std::move(v.view())) {}
    ZipperBase(ZipperBase&& v) = default;

    ZipperBase(const Derived& v) : ZipperBase(v.view()) {}
    ZipperBase(const View& v) : m_view(v) {}
    ZipperBase(const ZipperBase& v) = default;
    // Derived& operator=(concepts::ViewDerived auto const& v) {
    //     m_view = v;
    //     return derived();
    // }

    operator value_type() const
        requires(extents_type::rank() == 0)
    {
        return (*this)();
    }

    template <concepts::ViewDerived Other>
    ZipperBase(const Other& other)
        requires(view_type::is_writable &&
                 detail::extents::assignable_extents_v<
                     typename Other::extents_type, extents_type>)
        : m_view(extents_traits::convert_from(other.extents())) {
        m_view.assign(other);
    }

    template <typename... Args>
    ZipperBase(Args&&... args)
        : ZipperBase(View(std::forward<Args>(args)...)) {}

    template <concepts::ViewDerived Other>
    Derived& operator=(const Other& other)
        requires(view_type::is_writable &&
                 detail::extents::assignable_extents_v<
                     typename Other::extents_type, extents_type>)
    {
        m_view.assign(other);
        return derived();
    }
    template <concepts::ViewDerived Other>
    Derived& operator=(Other&& other)
        requires(view_type::is_writable &&
                 detail::extents::assignable_extents_v<
                     typename Other::extents_type, extents_type>)
    {
        m_view.assign(other);
        return derived();
    }
    // template <concepts::ZipperBaseDerived Other>
    // Derived& operator=(const Other& other)
    //     requires(view_type::is_writable &&
    //              detail::extents::assignable_extents_v<
    //                  typename Other::extents_type, extents_type>)
    //{
    //     m_view.assign(other.view());
    //     return derived();
    // }
    // template <concepts::ZipperBaseDerived Other>
    // Derived& operator=(Other&& other)
    //     requires(view_type::is_writable &&
    //              detail::extents::assignable_extents_v<
    //                  typename Other::extents_type, extents_type>)
    //{
    //     m_view.assign(other.view());
    //     return derived();
    // }

    template <concepts::ZipperBaseDerived Other>
    Derived& operator+=(const Other& other)
        requires(view_type::is_writable)
    {
        *this = *this + other;
    }
    template <concepts::ZipperBaseDerived Other>
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
    template <concepts::SliceLike... Slices>
    auto slice_view(Slices&&... slices) const {
        using view_type =
            views::unary::SliceView<const view_type, std::decay_t<Slices>...>;

        return view_type(view(), std::forward<Slices>(slices)...);
    }

    template <concepts::SliceLike... Slices>
    auto slice_view() const {
        using view_type =
            views::unary::SliceView<const view_type, std::decay_t<Slices>...>;
        return view_type(view(), Slices{}...);
    }

    template <concepts::SliceLike... Slices>
    auto slice_view(Slices&&... slices) {
        using view_type =
            views::unary::SliceView<view_type, std::decay_t<Slices>...>;
        return view_type(view(), std::forward<Slices>(slices)...);
    }
    template <concepts::SliceLike... Slices>
    auto slice_view() {
        using view_type =
            views::unary::SliceView<view_type, std::decay_t<Slices>...>;
        return view_type(view(), Slices{}...);
    }

   public:
   private:
    View m_view;
};

// TODO: figure out how to activate common declarations here rather than within
// each base type
// UNARY_DECLARATION(ZipperBase, LogicalNot, operator!)
// UNARY_DECLARATION(ZipperBase, BitNot, operator~)
// UNARY_DECLARATION(ZipperBase, Negate, operator-)
//
// SCALAR_BINARY_DECLARATION(ZipperBase, Plus, operator+)
// SCALAR_BINARY_DECLARATION(ZipperBase, Minus, operator-)
// SCALAR_BINARY_DECLARATION(ZipperBase, Multiplies, operator*)
// SCALAR_BINARY_DECLARATION(ZipperBase, Divides, operator/)
// SCALAR_BINARY_DECLARATION(ZipperBase, Modulus, operator%)
// SCALAR_BINARY_DECLARATION(ZipperBase, EqualsTo, operator==)
// SCALAR_BINARY_DECLARATION(ZipperBase, NotEqualsTo, operator!=)
// SCALAR_BINARY_DECLARATION(ZipperBase, Greater, operator>)
// SCALAR_BINARY_DECLARATION(ZipperBase, Less, operator<)
// SCALAR_BINARY_DECLARATION(ZipperBase, GreaterEqual, operator>=)
// SCALAR_BINARY_DECLARATION(ZipperBase, LessEqual, operator<=)
// SCALAR_BINARY_DECLARATION(ZipperBase, LogicalAnd, operator&&)
// SCALAR_BINARY_DECLARATION(ZipperBase, LogicalOr, operator||)
// SCALAR_BINARY_DECLARATION(ZipperBase, BitAnd, operator&)
// SCALAR_BINARY_DECLARATION(ZipperBase, BitOr, operator|)
// SCALAR_BINARY_DECLARATION(ZipperBase, BitXor, operator^)

// BINARY_DECLARATION(ZipperBase, Plus, operator+)
// BINARY_DECLARATION(ZipperBase, Minus, operator-)
//  BINARY_DECLARATION(ZipperBase, Divides, operator/)
//  BINARY_DECLARATION(ZipperBase, Modulus, operator%)
// BINARY_DECLARATION(ZipperBase, EqualsTo, operator==)
// BINARY_DECLARATION(ZipperBase, NotEqualsTo, operator!=)
//  BINARY_DECLARATION(ZipperBase, Greater, operator>)
//  BINARY_DECLARATION(ZipperBase, Less, operator<)
//  BINARY_DECLARATION(ZipperBase, GreaterEqual, operator>=)
//  BINARY_DECLARATION(ZipperBase, LessEqual, operator<=)
//  BINARY_DECLARATION(ZipperBase, LogicalAnd, operator&&)
//  BINARY_DECLARATION(ZipperBase, LogicalOr, operator||)
//  BINARY_DECLARATION(ZipperBase, BitAnd, operator&)
//  BINARY_DECLARATION(ZipperBase, BitOr, operator|)
//  BINARY_DECLARATION(ZipperBase, BitXor, operator^)

// template <concepts::ZipperBaseDerived View1, concepts::ZipperBaseDerived
// View2> bool operator==(View1 const& lhs, View2 const& rhs) {
//     return (lhs.as_array() == rhs.as_array()).all();
// }

// template <concepts::ZipperBaseDerived View1, concepts::ZipperBaseDerived
// View2> bool operator!=(View1 const& lhs, View2 const& rhs) {
//     return (lhs.as_array() != rhs.as_array()).any();
// }
}  // namespace zipper

#endif
