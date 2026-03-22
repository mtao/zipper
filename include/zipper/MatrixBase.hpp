#if !defined(ZIPPER_MATRIXBASE_HPP)
#define ZIPPER_MATRIXBASE_HPP

#include "ZipperBase.hpp"
#include "as.hpp"
#include "concepts/Index.hpp"
#include "concepts/IndexSlice.hpp"
#include "concepts/Matrix.hpp"
#include "concepts/Vector.hpp"
#include "concepts/detail/IsZipperBase.hpp"
#include "concepts/stl.hpp"
#include "zipper/types.hpp"
//
#include "concepts/DirectSolver.hpp"
#include "expression/nullary/StlMDArray.hpp"
#include "expression/reductions/Trace.hpp"
#include "expression/unary/TriangularView.hpp"
#include "zipper/detail/PartialReductionDispatcher.hpp"
#include "zipper/detail/constexpr_arithmetic.hpp"
#include "zipper/detail/extents/get_extent.hpp"

namespace zipper {
template<concepts::Expression Expr>
class ArrayBase;
template<typename ValueType, index_type Rows, index_type Cols, bool RowMajor = true>
class Matrix;

template<concepts::Expression Expr>
    requires(concepts::QualifiedRankedExpression<Expr, 2>)
class MatrixBase : public ZipperBase<MatrixBase, Expr> {
  public:
    MatrixBase() = default;

    using Base = ZipperBase<MatrixBase, Expr>;
    using expression_type = typename Base::expression_type;
    using expression_traits = typename Base::expression_traits;
    using value_type = typename expression_traits::value_type;
    using extents_type = typename expression_traits::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;

    using Base::Base;
    // using Base::operator=;
    using Base::cast;
    using Base::extent;
    using Base::extents;
    using Base::expression;
    using Base::swizzle;

    template<typename... Args>
        requires(!(concepts::Matrix<Args> && ...))
    MatrixBase(Args &&...args)
      : Base(std::in_place, std::forward<Args>(args)...) {}

    auto eval() const { return Matrix(*this); }

    template<concepts::Matrix Other>
    MatrixBase(const Other &other) : MatrixBase(other.expression()) {}

    auto operator=(concepts::Matrix auto const &v) -> MatrixBase & {
        return operator=(v.expression());
    }

    template<concepts::Expression Other>
    auto operator=(const Other &other) -> MatrixBase &
        requires(expression::concepts::WritableExpression<expression_type>)
    {
        Base::operator=(other);
        return *this;
    }

    void resize(index_type rows, index_type cols) {
        constexpr static bool dynamic_row = extents_traits::is_dynamic_extent(0);
        constexpr static bool dynamic_col = extents_traits::is_dynamic_extent(1);
        if constexpr (dynamic_row && dynamic_col) {
            expression().resize(extents_type{ rows, cols });
        } else if constexpr (dynamic_row) {
            expression().resize(extents_type{ rows });
        } else if constexpr (dynamic_col) {
            expression().resize(extents_type{ cols });
        }
    }
    void resize(index_type size)
        requires(extents_traits::rank_dynamic == 1)
    {
        expression().resize(extents_type{ size });
    }

    auto as_array() const { return zipper::as_array(*this); }

    auto trace() const -> value_type {
        return expression::reductions::Trace(expression())();
    }

    template<typename... Args, typename Self>
    auto operator()(this Self &&self, Args &&...idxs) -> decltype(auto) {
        if constexpr (concepts::IndexPack<std::decay_t<Args>...>) {
            if constexpr (std::is_const_v<std::remove_reference_t<Self>> || !expression::concepts::WritableExpression<expression_type>) {
                return std::as_const(self).expression()(
                  Base::filter_args_for_zipperbase(std::forward<Args>(idxs))...);
            } else {
                return self.expression()(
                  Base::filter_args_for_zipperbase(std::forward<Args>(idxs))...);
            }
        } else {
            using child_t = detail::member_child_storage_t<Self, expression_type>;
            using R = expression::unary::Slice<child_t,
                                               detail::slice_type_for_t<std::decay_t<Args>>...>;
            if constexpr (R::extents_type::rank() == 1) {
                return VectorBase<R>(std::in_place, std::forward<Self>(self).expression(), Base::filter_args_for_zipperbase(std::forward<Args>(idxs))...);
            } else if constexpr (R::extents_type::rank() == 2) {
                return MatrixBase<R>(std::in_place, std::forward<Self>(self).expression(), Base::filter_args_for_zipperbase(std::forward<Args>(idxs))...);
            }
        }
    }

  public:
    [[nodiscard]] constexpr auto rows() const -> index_type { return extent(0); }
    [[nodiscard]] constexpr auto cols() const -> index_type { return extent(1); }

    template<typename Slice, typename Self>
    auto row(this Self &&self) { return std::forward<Self>(self).template slice<Slice, full_extent_t>(); }
    template<typename Slice, typename Self>
    auto col(this Self &&self) { return std::forward<Self>(self).template slice<full_extent_t, Slice>(); }

    template<typename Slice, typename Self>
    auto row(this Self &&self, Slice &&s) {
        return std::forward<Self>(self).template slice<Slice, full_extent_t>(std::forward<Slice>(s), full_extent_t{});
    }
    template<typename Slice, typename Self>
    auto col(this Self &&self, Slice &&s) {
        return std::forward<Self>(self).template slice<full_extent_t, Slice>(full_extent_t{}, std::forward<Slice>(s));
    }
    template<typename... Slices, typename Self>
    auto slice(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, expression_type>;
        using V = expression::unary::Slice<child_t, std::decay_t<Slices>...>;
        if constexpr (V::extents_type::rank() == 2) {
            return MatrixBase<V>(std::in_place, std::forward<Self>(self).expression(), Slices{}...);
        } else {
            static_assert(V::extents_type::rank() == 1);
            return VectorBase<V>(std::in_place, std::forward<Self>(self).expression(), Slices{}...);
        }
    }

    template<typename... Slices, typename Self>
    auto slice(this Self &&self, Slices &&...slices) {
        using child_t = detail::member_child_storage_t<Self, expression_type>;
        using V = expression::unary::Slice<child_t,
                                           detail::slice_type_for_t<std::decay_t<Slices>>...>;
        if constexpr (V::extents_type::rank() == 2) {
            return MatrixBase<V>(std::in_place, std::forward<Self>(self).expression(), Base::filter_args_for_zipperbase(std::forward<Slices>(slices))...);
        } else {
            static_assert(V::extents_type::rank() == 1);
            return VectorBase<V>(std::in_place, std::forward<Self>(self).expression(), Base::filter_args_for_zipperbase(std::forward<Slices>(slices))...);
        }
    }

    template<rank_type... ranks, typename Self>
    auto swizzle(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, expression_type>;
        using V = expression::unary::Swizzle<child_t, ranks...>;
        return MatrixBase<V>(std::in_place, std::forward<Self>(self).expression());
    }
    template<typename Self>
    auto transpose(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, expression_type>;
        using V = expression::unary::Swizzle<child_t, 1, 0>;
        return MatrixBase<V>(std::in_place, std::forward<Self>(self).expression());
    }

    /// @brief Returns a read-only triangular view of this matrix.
    ///
    /// @tparam Mode  Compile-time TriangularMode flag (e.g. Lower, Upper,
    ///               UnitLower, UnitUpper, StrictlyLower, StrictlyUpper).
    /// @return A MatrixBase wrapping a TriangularView expression.
    template<expression::TriangularMode Mode, typename Self>
    auto as_triangular(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, expression_type>;
        using ViewType = expression::unary::TriangularView<Mode, child_t>;
        return MatrixBase<ViewType>(std::in_place, std::forward<Self>(self).expression());
    }

    /// @brief Solve A * x = b, forwarding to the underlying expression's
    ///        `.solve()` method.
    ///
    /// This overload is only available when the underlying expression type
    /// satisfies the `DirectSolver` concept (e.g. when this MatrixBase wraps
    /// a TriangularView, or a decomposition result).
    template<concepts::Vector BDerived>
        requires(concepts::DirectSolver<expression_type>)
    auto solve(const BDerived &b) const {
        return expression().solve(b);
    }

    template<concepts::IndexSlice Slice, typename Self>
    auto row_slice(this Self &&self, const Slice &s = {}) {
        return std::forward<Self>(self).slice(s, full_extent_t{});
    }
    template<concepts::IndexSlice Slice, typename Self>
    auto col_slice(this Self &&self, const Slice &s = {}) {
        return std::forward<Self>(self).slice(full_extent_t{}, s);
    }

    // meh names for alignment with eigen
    template<concepts::Index Index, typename Self>
    auto topRows(this Self &&self, const Index &s) {
        return std::forward<Self>(self).row_slice(zipper::slice({}, s));
    }
    template<concepts::Index Index, typename Self>
    auto leftCols(this Self &&self, const Index &s) {
        return std::forward<Self>(self).col_slice(zipper::slice({}, s));
    }

    template<concepts::Index Index, typename Self>
    auto bottomRows(this Self &&self, const Index &s) {
        auto end = detail::extents::get_extent<0>(self.extents());
        detail::ConstexprArithmetic size(s);
        return std::forward<Self>(self).row_slice(zipper::slice((end - size).value(), s));
    }
    template<concepts::Index Index, typename Self>
    auto rightCols(this Self &&self, const Index &s) {
        auto end = detail::extents::get_extent<1>(self.extents());
        detail::ConstexprArithmetic size(s);
        return std::forward<Self>(self).col_slice(zipper::slice((end - size).value(), s));
    }
    template<index_type Size, typename Self>
    auto topRows(this Self &&self) {
        return std::forward<Self>(self).row_slice(zipper::slice({}, static_index_t<Size>{}));
    }
    template<index_type Size, typename Self>
    auto leftCols(this Self &&self) {
        return std::forward<Self>(self).col_slice(zipper::slice({}, static_index_t<Size>{}));
    }

    template<index_type Size, typename Self>
    auto bottomRows(this Self &&self) {
        detail::ConstexprArithmetic size(static_index_t<Size>{});
        auto t = detail::extents::get_extent<0>(self.extents()) - size;
        return std::forward<Self>(self).row_slice(zipper::slice(t.value(), size.value()));
    }
    template<index_type Size, typename Self>
    auto rightCols(this Self &&self) {
        detail::ConstexprArithmetic size(static_index_t<Size>{});
        auto t = detail::extents::get_extent<1>(self.extents()) - size;
        return std::forward<Self>(self).col_slice(zipper::slice(t.value(), size.value()));
    }

    template<typename Self>
    auto rowwise(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, expression_type>;
        // we're reducing the first cols
        return detail::PartialReductionDispatcher<VectorBase, child_t, 1>(
          std::forward<Self>(self).expression());
    }
    template<typename Self>
    auto colwise(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, expression_type>;
        // we're reducing the first rows
        return detail::PartialReductionDispatcher<VectorBase, child_t, 0>(
          std::forward<Self>(self).expression());
    }
};

template<typename T>
using MatrixXX = Matrix<T, dynamic_extent, dynamic_extent>;
template<typename T, index_type R>
using MatrixRX = Matrix<T, R, dynamic_extent>;
template<typename T, index_type C>
using MatrixXC = Matrix<T, dynamic_extent, C>;

template<concepts::Expression Expr>
MatrixBase(Expr &&) -> MatrixBase<Expr>;
template<concepts::Expression Expr>
MatrixBase(const Expr &) -> MatrixBase<Expr>;

// STL deduction guides: rvalue → owning StlMDArray, lvalue → borrowing StlMDArray
template<concepts::StlStorageOfRank<2> S>
MatrixBase(S &&) -> MatrixBase<expression::nullary::StlMDArray<std::decay_t<S>>>;
template<concepts::StlStorageOfRank<2> S>
MatrixBase(S &) -> MatrixBase<expression::nullary::StlMDArray<S &>>;

namespace concepts::detail {
    template<typename T>
    struct IsMatrix<MatrixBase<T>> : std::true_type {};
    template<typename T>
    struct IsZipperBase<MatrixBase<T>> : std::true_type {};
}// namespace concepts::detail

}// namespace zipper
#endif
