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
#include "expression/nullary/StlMDArray.hpp"
#include "expression/reductions/Trace.hpp"
#include "expression/unary/Diagonal.hpp"
#include "zipper/detail/PartialReductionDispatcher.hpp"
#include "zipper/detail/constexpr_arithmetic.hpp"
#include "zipper/detail/extents/get_extent.hpp"

namespace zipper {
template <concepts::Expression Expr> class ArrayBase;
template <typename ValueType, index_type Rows, index_type Cols,
          bool RowMajor = true>
class Matrix;

template <concepts::Expression Expr>
class MatrixBase : public ZipperBase<MatrixBase, Expr> {
public:
  MatrixBase() = default;

  using Base = ZipperBase<MatrixBase, Expr>;
  using expression_type = typename Base::expression_type;
  using expression_traits = typename Base::expression_traits;
  using value_type = typename expression_traits::value_type;
  using extents_type = typename expression_traits::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;
  static_assert(extents_traits::rank == 2);

  using Base::Base;
  // using Base::operator=;
  using Base::cast;
  using Base::extent;
  using Base::extents;
  using Base::expression;
  using Base::swizzle;

  template <typename... Args>
    requires(!(concepts::Matrix<Args> && ...))
  MatrixBase(Args &&...args)
      : Base(std::in_place, std::forward<Args>(args)...) {}

  auto eval() const { return Matrix(*this); }

  template <concepts::Matrix Other>
  MatrixBase(const Other &other) : MatrixBase(other.expression()) {}

  auto operator=(concepts::Matrix auto const &v) -> MatrixBase & {
    return operator=(v.expression());
  }

  template <concepts::Expression Other>
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
      expression().resize(extents_type{rows, cols});
    } else if constexpr (dynamic_row) {
      expression().resize(extents_type{rows});
    } else if constexpr (dynamic_col) {
      expression().resize(extents_type{cols});
    }
  }
  void resize(index_type size)
    requires(extents_traits::rank_dynamic == 1)
  {
    expression().resize(extents_type{size});
  }

  auto as_array() const { return zipper::as_array(*this); }

  auto trace() const -> value_type {
    return expression::reductions::Trace(expression())();
  }

  template <typename... Args>
  auto operator()(Args &&...idxs) const -> decltype(auto)

  {
    if constexpr (concepts::IndexPack<std::decay_t<Args>...>) {
      return Base::operator()(std::forward<Args>(idxs)...);
    } else {
      using R = expression::unary::Slice<const expression_type&, std::decay_t<Args>...>;
      if constexpr (R::extents_type::rank() == 1) {
        return VectorBase<R>(std::in_place, expression(), std::forward<Args>(idxs)...);
      } else if constexpr (R::extents_type::rank() == 2) {
        return MatrixBase<R>(std::in_place, expression(), std::forward<Args>(idxs)...);
      }
    }
  }
  template <typename... Args>
  auto operator()(Args &&...idxs) -> decltype(auto)

  {
    if constexpr (concepts::IndexPack<std::decay_t<Args>...>) {
      return Base::operator()(std::forward<Args>(idxs)...);
    } else {
      using R = expression::unary::Slice<expression_type&, std::decay_t<Args>...>;
      if constexpr (R::extents_type::rank() == 1) {
        return VectorBase<R>(std::in_place, expression(), std::forward<Args>(idxs)...);
      } else if constexpr (R::extents_type::rank() == 2) {
        return MatrixBase<R>(std::in_place, expression(), std::forward<Args>(idxs)...);
      }
    }
  }

public:
  [[nodiscard]] constexpr auto rows() const -> index_type { return extent(0); }
  [[nodiscard]] constexpr auto cols() const -> index_type { return extent(1); }

  template <typename Slice> auto row() { return slice<Slice, full_extent_t>(); }
  template <typename Slice> auto col() { return slice<full_extent_t, Slice>(); }

  template <typename Slice> auto row() const {
    return slice<Slice, full_extent_t>();
  }
  template <typename Slice> auto col() const {
    return slice<full_extent_t, Slice>();
  }

  template <typename Slice> auto row(Slice &&s) {
    return slice<Slice, full_extent_t>(std::forward<Slice>(s), full_extent_t{});
  }
  template <typename Slice> auto col(Slice &&s) {
    return slice<full_extent_t, Slice>(full_extent_t{}, std::forward<Slice>(s));
  }

  template <typename Slice> auto row(Slice &&s) const {
    return slice<Slice, full_extent_t>(std::forward<Slice>(s), full_extent_t{});
  }
  template <typename Slice> auto col(Slice &&s) const {
    return slice<full_extent_t, Slice>(full_extent_t{}, std::forward<Slice>(s));
  }
  template <typename... Slices> auto slice() {
    using V = expression::unary::Slice<expression_type&, std::decay_t<Slices>...>;
    if constexpr (V::extents_type::rank() == 2) {
      return MatrixBase<V>(std::in_place, expression(), Slices{}...);
    } else {
      static_assert(V::extents_type::rank() == 1);
      return VectorBase<V>(std::in_place, expression(), Slices{}...);
    }
  }

  template <typename... Slices> auto slice(Slices &&...slices) const {
    using V = expression::unary::Slice<const expression_type&, std::decay_t<Slices>...>;
    if constexpr (V::extents_type::rank() == 2) {
      return MatrixBase<V>(std::in_place, expression(), std::forward<Slices>(slices)...);
    } else {
      static_assert(V::extents_type::rank() == 1);
      return VectorBase<V>(std::in_place, expression(), std::forward<Slices>(slices)...);
    }
  }
  template <typename... Slices> auto slice() const {
    using V = expression::unary::Slice<const expression_type&, std::decay_t<Slices>...>;
    if constexpr (V::extents_type::rank() == 2) {
      return MatrixBase<V>(std::in_place, expression(), Slices{}...);
    } else {
      static_assert(V::extents_type::rank() == 1);
      return VectorBase<V>(std::in_place, expression(), Slices{}...);
    }
  }

  template <typename... Slices> auto slice(Slices &&...slices) {
    using V = expression::unary::Slice<expression_type&, std::decay_t<Slices>...>;
    if constexpr (V::extents_type::rank() == 2) {
      return MatrixBase<V>(std::in_place, expression(), std::forward<Slices>(slices)...);
    } else {
      static_assert(V::extents_type::rank() == 1);
      return VectorBase<V>(std::in_place, expression(), std::forward<Slices>(slices)...);
    }
  }

  auto diagonal() const {
    return VectorBase<expression::unary::Diagonal<const expression_type&>>(std::in_place, expression());
  }
  auto diagonal() {
    return VectorBase<expression::unary::Diagonal<expression_type&>>(std::in_place, expression());
  }

  template <rank_type... ranks> auto swizzle() const {
    return Base::template swizzle<MatrixBase, ranks...>();
  }
  template <rank_type... ranks> auto swizzle() {
    return Base::template swizzle<MatrixBase, ranks...>();
  }
  auto transpose() const { return Base::template swizzle<MatrixBase, 1, 0>(); }
  auto transpose() { return Base::template swizzle<MatrixBase, 1, 0>(); }

  template <concepts::IndexSlice Slice> auto row_slice(const Slice &s = {}) {
    return slice(s, full_extent_t{});
  }
  template <concepts::IndexSlice Slice>
  auto row_slice(const Slice &s = {}) const {
    return slice(s, full_extent_t{});
  }
  template <concepts::IndexSlice Slice> auto col_slice(const Slice &s = {}) {
    return slice(full_extent_t{}, s);
  }
  template <concepts::IndexSlice Slice>
  auto col_slice(const Slice &s = {}) const {
    return slice(full_extent_t{}, s);
  }

  // meh names for alignment with eigen
  template <concepts::Index Index> auto topRows(const Index &s) {
    return row_slice(zipper::slice({}, s));
  }
  template <concepts::Index Index> auto topRows(const Index &s) const {
    return row_slice(zipper::slice({}, s));
  }
  template <concepts::Index Index> auto leftCols(const Index &s) {
    return col_slice(zipper::slice({}, s));
  }
  template <concepts::Index Index> auto leftCols(const Index &s) const {
    return col_slice(zipper::slice({}, s));
  }

  template <concepts::Index Index> auto bottomRows(const Index &s) {
    auto end = detail::extents::get_extent<0>(extents());
    detail::ConstexprArithmetic size(s);
    return row_slice(zipper::slice((end - size).value(), s));
  }
  template <concepts::Index Index> auto bottomRows(const Index &s) const {
    auto end = detail::extents::get_extent<0>(extents());
    detail::ConstexprArithmetic size(s);
    return row_slice(zipper::slice((end - size).value(), s));
  }
  template <concepts::Index Index> auto rightCols(const Index &s) {
    auto end = detail::extents::get_extent<1>(extents());
    detail::ConstexprArithmetic size(s);
    return col_slice(zipper::slice((end - size).value(), s));
  }
  template <concepts::Index Index> auto rightCols(const Index &s) const {
    auto end = detail::extents::get_extent<1>(extents());
    detail::ConstexprArithmetic size(s);
    return col_slice(zipper::slice((end - size).value(), s));
  }
  template <index_type Size> auto topRows() {
    return row_slice(zipper::slice({}, static_index_t<Size>{}));
  }
  template <index_type Size> auto topRows() const {
    return row_slice(zipper::slice({}, static_index_t<Size>{}));
  }
  template <index_type Size> auto leftCols() {
    return col_slice(zipper::slice({}, static_index_t<Size>{}));
  }
  template <index_type Size> auto leftCols() const {
    return col_slice(zipper::slice({}, static_index_t<Size>{}));
  }

  template <index_type Size> auto bottomRows() {
    detail::ConstexprArithmetic size(static_index_t<Size>{});
    auto t = size - detail::extents::get_extent<0>(extents());
    return row_slice(zipper::slice(t.value(), size.value()));
  }
  template <index_type Size> auto bottomRows() const {
    detail::ConstexprArithmetic size(static_index_t<Size>{});
    auto t = size - detail::extents::get_extent<0>(extents());
    return row_slice(zipper::slice(t.value(), size.value()));
  }
  template <index_type Size> auto rightCols() {
    detail::ConstexprArithmetic size(static_index_t<Size>{});
    auto t = size - detail::extents::get_extent<1>(extents());
    return col_slice(zipper::slice(t.value(), size.value()));
  }
  template <index_type Size> auto rightCols() const {
    detail::ConstexprArithmetic size(static_index_t<Size>{});
    auto t = size - detail::extents::get_extent<1>(extents());
    return col_slice(zipper::slice(t.value(), size.value()));
  }

  auto rowwise() {
    // we're reducing the first cols
    return detail::PartialReductionDispatcher<VectorBase, expression_type&, 1>(expression());
  }
  auto colwise() {
    // we're reducing the first rows
    return detail::PartialReductionDispatcher<VectorBase, expression_type&, 0>(expression());
  }
  auto rowwise() const {
    // we're reducing the first cols
    return detail::PartialReductionDispatcher<VectorBase, const expression_type&, 1>(
        expression());
  }
  auto colwise() const {
    // we're reducing the first rows
    return detail::PartialReductionDispatcher<VectorBase, const expression_type&, 0>(
        expression());
  }
};

template <typename T>
using MatrixXX = Matrix<T, dynamic_extent, dynamic_extent>;
template <typename T, index_type R>
using MatrixRX = Matrix<T, R, dynamic_extent>;
template <typename T, index_type C>
using MatrixXC = Matrix<T, dynamic_extent, C>;

template <concepts::Expression Expr>
MatrixBase(Expr &&) -> MatrixBase<Expr>;
template <concepts::Expression Expr>
MatrixBase(const Expr &) -> MatrixBase<Expr>;

// STL deduction guides: rvalue → owning StlMDArray, lvalue → borrowing StlMDArray
template <concepts::StlStorageOfRank<2> S>
MatrixBase(S &&) -> MatrixBase<expression::nullary::StlMDArray<std::decay_t<S>>>;
template <concepts::StlStorageOfRank<2> S>
MatrixBase(S &) -> MatrixBase<expression::nullary::StlMDArray<S &>>;

namespace concepts::detail {
template <typename T>
struct IsMatrix<MatrixBase<T>> : std::true_type {};
template <typename T>
struct IsZipperBase<MatrixBase<T>> : std::true_type {};
} // namespace concepts::detail

} // namespace zipper
#endif
