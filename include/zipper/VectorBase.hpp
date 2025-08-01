#include "zipper/views/unary/concepts/ScalarOperation.hpp"
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
#include "detail/constexpr_arithmetic.hpp"
#include "detail/extents/constexpr_extent.hpp"
#include "views/binary/CrossProductView.hpp"
#include "views/reductions/CoefficientSum.hpp"
#include "views/unary/HomogeneousView.hpp"

namespace zipper {
template <typename ValueType, index_type Rows>
class Vector;

template <concepts::QualifiedViewDerived View>
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
    static_assert(std::is_same_v<extents_type, typename Base::extents_type>);

    using Base::Base;
    // using Base::operator=;
    using Base::cast;
    using Base::extent;
    using Base::extents;
    using Base::swizzle;
    using Base::view;

    auto eval() const { return Vector(*this); }
    template <concepts::VectorBaseDerived Other>
    VectorBase(const Other& other)
        requires(view_type::is_writable)
        : VectorBase(other.view()) {}

    VectorBase(concepts::QualifiedViewDerived auto& v) : Base(v) {}
    VectorBase(concepts::QualifiedViewDerived auto&& v) : Base(std::move(v)) {}
    VectorBase(const extents_type& e) : Base(e) {}
    VectorBase& operator=(concepts::QualifiedViewDerived auto const& v) {
        return Base::operator=(v);
    }
    template <typename... Args>
    VectorBase(Args&&... args)
        : VectorBase(View(std::forward<Args>(args)...)) {}

    template <concepts::VectorBaseDerived Other>
    VectorBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        view().assign(other.view());
        return *this;
    }
    template <concepts::VectorBaseDerived Other>
    VectorBase& operator=(Other&& other)
        requires(view_type::is_writable)
    {
        return operator=(other.view());
    }
    constexpr index_type size() const { return extent(0); }
    constexpr index_type rows() const { return extent(0); }

    void resize(index_type size)
        requires(extents_traits::is_dynamic)
    {
        view().resize(extents_type{size});
    }

    // TODO: make vectorbase or zipperbase assignable from initializer lists
    // template <typename T>
    // VectorBase& operator=(const std::initializer_list<T>& l)
    //    requires(extents_traits::is_dynamic)
    //{
    //    view().resize(extents_type(l.size()));
    //    std::ranges::copy(l, begin());
    //}

    template <typename T>
    VectorBase& operator=(const std::initializer_list<T>& l)
        requires(extents_traits::is_static)
    {
        assert(l.size() == extent(0));
        for (index_type j = 0; j < extent(0); ++j) {
            (*this)(j) = std::data(l)[j];
        }
        return *this;
    }
    template <typename T>
    VectorBase& operator=(const std::initializer_list<T>& l)
        requires(extents_traits::is_dynamic)
    {
        view().resize(extents_type(l.size()));
        for (index_type j = 0; j < extent(0); ++j) {
            (*this)(j) = std::data(l)[j];
        }
        return *this;
    }

    // auto as_col_matrix() const {
    //     return MatrixBase<views::unary::IdentityView<View>>(view());
    // }
    auto as_array() const { return zipper::as_array(*this); }
    auto as_tensor() const { return zipper::as_tensor(*this); }
    auto as_form() const { return zipper::as_form(*this); }

    template <index_type T>
    value_type norm_powered() const {
        return views::reductions::LpNormPowered<T, const view_type>(view())();
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

    template <index_type Start, index_type Size>
    auto segment()
        requires(view_type::is_writable)
    {
        auto S = slice(std::integral_constant<index_type, Start>{},
                       std::integral_constant<index_type, Size>{});
        auto v = Base::slice_view(S);
        using V = std::decay_t<decltype(v)>;
        return VectorBase<V>(std::move(v));
    }
    template <index_type Start, index_type Size>
    auto segment() const {
        auto S = slice(std::integral_constant<index_type, Start>{},
                       std::integral_constant<index_type, Size>{});
        auto v = Base::slice_view(S);
        using V = std::decay_t<decltype(v)>;
        return VectorBase<V>(std::move(v));
    }
    template <index_type Size>
    auto segment(index_type start)
        requires(view_type::is_writable)
    {
        auto S = slice(start, std::integral_constant<index_type, Size>{});
        auto v = Base::slice_view(S);
        using V = std::decay_t<decltype(v)>;
        return VectorBase<V>(std::move(v));
    }
    template <index_type Size>
    auto segment(index_type start) const {
        auto S = slice(start, Size);
        auto v = Base::slice_view(S);
        using V = std::decay_t<decltype(v)>;
        return VectorBase<V>(std::move(v));
    }
    auto segment(index_type start, index_type size)
        requires(view_type::is_writable)
    {
        auto S = slice(start, size);
        auto v = Base::slice_view(S);
        using V = std::decay_t<decltype(v)>;
        return VectorBase<V>(std::move(v));
    }
    auto segment(index_type start, index_type size) const {
        auto S = slice(start, size);
        auto v = Base::slice_view(S);
        using V = std::decay_t<decltype(v)>;
        return VectorBase<V>(std::move(v));
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

    auto get_tail_slice(index_type I) const {
        return slice(
            detail::minus(detail::extents::constexpr_extent<0>(extents()), I),
            I);
    }
    template <index_type I>
    auto get_tail_slice() const {
        return slice(
            detail::minus(detail::extents::constexpr_extent<0>(extents()),
                          std::integral_constant<index_type, I>{}),
            std::integral_constant<index_type, I>{});
    }
    template <index_type I>
    auto tail()
        requires(view_type::is_writable)
    {
        auto S = get_tail_slice<I>();
        auto v = Base::slice_view(S);
        using V = std::decay_t<decltype(v)>;
        return VectorBase<V>(std::move(v));
    }
    template <index_type I>
    auto tail() const {
        auto S = get_tail_slice<I>();
        auto v = Base::slice_view(S);
        using V = std::decay_t<decltype(v)>;
        return VectorBase<V>(std::move(v));
    }
    auto tail(index_type N)
        requires(view_type::is_writable)
    {
        auto S = get_tail_slice(N);
        auto v = Base::slice_view(S);
        using V = std::decay_t<decltype(v)>;
        return VectorBase<V>(std::move(v));
    }
    auto tail(index_type N) const {
        auto S = get_tail_slice(N);
        auto v = Base::slice_view(S);
        using V = std::decay_t<decltype(v)>;
        return VectorBase<V>(std::move(v));
    }

    // implements ones * this.transpose()
    auto repeat_left() const {
        return Base::template repeat_left<1, MatrixBase>();
    }
    // implements  this * ones.transpose()
    auto repeat_right() const {
        return Base::template repeat_right<1, MatrixBase>();
    }

    template <index_type T = 2>
    value_type norm() const {
        return views::reductions::LpNorm<T, const view_type>(view())();
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

template <class T, std::size_t Size = std::dynamic_extent>
VectorBase(std::span<T, Size> s)
    -> VectorBase<storage::SpanStorage<T, zipper::extents<Size>>>;

template <class T, std::size_t Size = std::dynamic_extent>
VectorBase(std::span<const T, Size> s)
    -> VectorBase<storage::SpanStorage<const T, zipper::extents<Size>>>;

template <class T, std::size_t Size = std::dynamic_extent>
VectorBase(const std::array<T, Size>& s)
    -> VectorBase<storage::SpanStorage<const T, zipper::extents<Size>>>;
template <class T, std::size_t Size = std::dynamic_extent>
VectorBase(std::array<T, Size>& s)
    -> VectorBase<storage::SpanStorage<T, zipper::extents<Size>>>;

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
    using V = views::unary::ScalarMultipliesView<
        typename View::value_type, const typename View::view_type, true>;
    return VectorBase<V>(V(lhs.view(), rhs));
}
template <concepts::VectorBaseDerived View>
auto operator*(typename View::value_type const& lhs, View const& rhs) {
    using V = views::unary::ScalarMultipliesView<
        typename View::value_type, const typename View::view_type, false>;
    return VectorBase<V>(V(lhs, rhs.view()));
}

}  // namespace zipper
#endif
