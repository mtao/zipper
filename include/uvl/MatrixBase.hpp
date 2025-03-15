#if !defined(UVL_MATRIXBASE_HPP)
#define UVL_MATRIXBASE_HPP

#include "ArrayBase.hpp"
#include "uvl/detail/convert_extents.hpp"
#include "uvl/types.hpp"
//
#include "concepts/MatrixBaseDerived.hpp"
#include "concepts/MatrixViewDerived.hpp"
#include "concepts/VectorBaseDerived.hpp"
//
#include "views/binary/AdditionView.hpp"
#include "views/unary/IdempotentView.hpp"
#include "views/binary/MatrixProductView.hpp"
#include "views/binary/MatrixVectorProductView.hpp"
#include "views/reductions/CoefficientSum.hpp"
#include "views/unary/CastView.hpp"
#include "views/unary/NegateView.hpp"
#include "views/unary/ScalarProductView.hpp"
#include "views/unary/ScalarQuotientView.hpp"
#include "views/unary/SwizzleView.hpp"

namespace uvl {
template <concepts::ViewDerived View>
class ArrayBase;

template <concepts::MatrixViewDerived View>
class MatrixBase {
   public:
    MatrixBase() = default;

    using view_type = View;
    using value_type = View::value_type;
    using extents_type = View::extents_type;
    auto eval() const {
        return Matrix(*this);
    }
    template <typename... Args>
    MatrixBase(Args&&... v) : m_view(std::forward<Args>(v)...) {}

    MatrixBase(View&& v) : m_view(v) {}
    MatrixBase(const View& v) : m_view(v) {}
    MatrixBase& operator=(View&& v) { m_view = v; }
    MatrixBase& operator=(const View& v) { m_view = v; }

    MatrixBase(MatrixBase&& v) = default;
    MatrixBase(const MatrixBase& v) = default;
    MatrixBase& operator=(MatrixBase&& v) = default;
    MatrixBase& operator=(const MatrixBase& v) = default;


    template <concepts::MatrixViewDerived Other>
    MatrixBase(const Other& other)
        requires(view_type::is_writable)
        : m_view(detail::convert_extents<extents_type>(other.extents())) {
        m_view.assign(other);
    }
    template <concepts::MatrixViewDerived Other>
    MatrixBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        m_view.assign(other);
        return *this;
    }
    template <concepts::MatrixBaseDerived Other>
    MatrixBase(const Other& other)
        requires(view_type::is_writable)
        : MatrixBase(other.view()) {}

    template <concepts::MatrixBaseDerived Other>
    MatrixBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        return operator=(other.view());
    }

    MatrixBase& operator*=(const value_type& other)
        requires(view_type::is_writable)
    {
        return *this = other * *this;
    }
    MatrixBase& operator/=(const value_type& other)
        requires(view_type::is_writable)
    {
        return *this = *this / other;
    }
    template <concepts::MatrixBaseDerived Other>
    MatrixBase& operator+=(const Other& other)
        requires(view_type::is_writable)
    {
        return *this = *this + other;
    }
    template <concepts::MatrixBaseDerived Other>
    MatrixBase& operator-=(const Other& other)
        requires(view_type::is_writable)
    {
        return *this = *this - other;
    }


    auto as_array() const { return ArrayBase<views::unary::IdempotentView<View>>(view()); }

    template <concepts::MatrixBaseDerived Other>
    friend auto operator+(const MatrixBase<view_type>& lhs, Other const& rhs) {
        return MatrixBase<
            views::binary::AdditionView<view_type, typename Other::view_type>>(
            lhs.view(), rhs.view());
    }

    friend auto operator-(const MatrixBase<view_type>& lhs) {
        return MatrixBase<views::unary::NegateView<view_type>>(lhs.view());
    }
    template <concepts::MatrixBaseDerived Other>
    friend auto operator-(const MatrixBase<view_type>& lhs, Other const& rhs) {
        return lhs + (-rhs);
    }
    template <concepts::MatrixBaseDerived Other>
    friend auto operator*(const MatrixBase<view_type>& lhs, Other const& rhs) {
        return MatrixBase<views::binary::MatrixProductView<
            view_type, typename Other::view_type>>(lhs.view(), rhs.view());
    }

    template <concepts::VectorBaseDerived Other>
    friend auto operator*(const MatrixBase<view_type>& lhs, Other const& rhs) {
        return VectorBase<views::binary::MatrixVectorProductView<
            view_type, typename Other::view_type>>(lhs.view(), rhs.view());
    }

    friend auto operator*(const MatrixBase<view_type>& lhs,
                          value_type const& rhs) {
        return MatrixBase<
            views::unary::ScalarProductView<value_type, view_type>>(rhs,
                                                                    lhs.view());
    }
    friend auto operator*(value_type const& lhs,
                          const MatrixBase<view_type>& rhs) {
        return MatrixBase<
            views::unary::ScalarProductView<value_type, view_type>>(lhs,
                                                                    rhs.view());
    }
    friend auto operator/(const MatrixBase<view_type>& lhs,
                          value_type const& rhs) {
        return MatrixBase<
            views::unary::ScalarQuotientView<view_type, value_type>>(lhs.view(),
                                                                     rhs);
    }

    template <typename T>
    auto cast() const {
        return MatrixBase<views::unary::CastView<T, view_type>>(view());
    }

    template <rank_type... ranks>
    auto swizzle() const {
        return MatrixBase<views::unary::SwizzleView<view_type, ranks...>>(
            views::unary::SwizzleView<view_type, ranks...>(view()));
    }
    auto transpose() const { return swizzle<1, 0>(); }

    /*
    template <index_type... Indices,
              typename mapping_type = std::experimental::layout_left>
    auto reshape(const uvl::extents<Indices...>& e,
                 mapping_type map = {}) const {
        if constexpr (sizeof...(Indices) == 1 &&
                      concepts::MappedViewDerived<view_type>) {
            return nullptr;
        } else if constexpr (extents_type::rank == 1) {
        }
        // return MatrixBase<views::unary::CastView<T, view_type>>(view());
    }
    */

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

template <concepts::MatrixViewDerived View>
MatrixBase(View&& view) -> MatrixBase<View>;
template <concepts::MatrixViewDerived View>
MatrixBase(const View& view) -> MatrixBase<View>;

}  // namespace uvl

// namespace uvl::views {

// template <typename ValueType, index_type Rows, index_type Cols>
// struct detail::ViewTraits<Matrix<ValueType, Rows, Cols>>
//     : public detail::ViewTraits<
//           uvl::storage::PlainObjectStorage<ValueType, extents<Rows, Cols>>>
//           {};
// }  // namespace uvl::views
// template <concepts::MatrixViewDerived View>
// struct detail::ViewTraits<MatrixBase<view_type>> : public
// detail::ViewTraits<View>
// {
// };
#endif
