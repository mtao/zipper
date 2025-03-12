#if !defined(UVL_MATRIXBASE_HPP)
#define UVL_MATRIXBASE_HPP

#include "detail/convert_extents.hpp"
#include "uvl/types.hpp"
//
#include "concepts/MatrixBaseDerived.hpp"
#include "concepts/MatrixViewDerived.hpp"
#include "concepts/VectorViewDerived.hpp"
#include "concepts/MappedViewDerived.hpp"
//
#include "views/binary/AdditionView.hpp"
#include "views/binary/MatrixProductView.hpp"
#include "views/binary/MatrixVectorProductView.hpp"
#include "views/unary/CastView.hpp"
#include "views/unary/ScalarProductView.hpp"

namespace uvl {

template <concepts::MatrixViewDerived View>
class MatrixBase {
   public:
    MatrixBase() = default;

    using view_type = View;
    using value_type = View::value_type;
    using extents_type = View::extents_type;
    template <typename... Args>
    MatrixBase(Args&&... v) : m_view(std::forward<Args>(v)...) {}

    template <concepts::MatrixBaseDerived Other>
    MatrixBase(const Other& other)
        : m_view(detail::convert_extents<extents_type>(other.extents())) {
        m_view.assign(other.view());
    }
    template <concepts::MatrixBaseDerived Other>
    MatrixBase& operator=(const Other& other) {
        m_view.assign(other.view());
    }

    template <concepts::MatrixBaseDerived Other>
    friend auto operator+(const MatrixBase<view_type>& lhs, Other const& rhs) {
        return MatrixBase<
            views::binary::AdditionView<view_type, typename Other::view_type>>(
            lhs.view(), rhs.view());
    }
    template <concepts::MatrixBaseDerived Other>
    friend auto operator*(const MatrixBase<view_type>& lhs, Other const& rhs) {
        return MatrixBase<views::binary::MatrixProductView<
            view_type, typename Other::view_type>>(lhs.view(), rhs.view());
    }

    template <concepts::MatrixBaseDerived Other>
    friend auto operator*(const MatrixBase<view_type>& lhs, Other const& rhs) 
    requires(concepts::VectorViewDerived<typename Other::view_type>)
    {
        return MatrixBase<views::binary::MatrixVectorProductView<
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

    template <typename T>
    auto cast() const {
        return MatrixBase<views::unary::CastView<T, view_type>>(view());
    }

    template <index_type... Indices, typename mapping_type = std::experimental::layout_left>
    auto reshape(const extents<Indices...>& e, mapping_type map = {}) const {
        if constexpr(sizeof...(Indices) == 1 && concepts::MappedViewDerived<view_type>)  {
            return nullptr;
        } else if constexpr(extents_type::rank == 1)  {

        }
        //return MatrixBase<views::unary::CastView<T, view_type>>(view());
    }

    template <typename... Args>
    value_type operator()(Args&&... idxs) const

    {
        return view()(std::forward<Args>(idxs)...);
    }

    const View& view() const { return m_view; }
    View& view() { return m_view; }
    const extents_type& extents() const { return view().extents(); }

   private:
    View m_view;
};

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
