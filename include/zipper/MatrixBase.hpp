#if !defined(ZIPPER_MATRIXBASE_HPP)
#define ZIPPER_MATRIXBASE_HPP

#include "ZipperBase.hpp"
#include "as.hpp"
#include "concepts/Index.hpp"
#include "concepts/IndexSlice.hpp"
#include "concepts/Matrix.hpp"
#include "concepts/Vector.hpp"
#include "concepts/detail/IsZipperBase.hpp"
#include "zipper/types.hpp"
//
#include "expression/binary/MatrixProduct.hpp"
#include "expression/binary/MatrixVectorProduct.hpp"
#include "expression/reductions/Trace.hpp"
#include "expression/unary/Diagonal.hpp"
#include "expression/unary/ScalarArithmetic.hpp"
#include "zipper/detail/PartialReductionDispatcher.hpp"
#include "zipper/detail/constexpr_arithmetic.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/detail/extents/get_extent.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"

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

  auto eval() const { return Matrix(*this); }

  template <concepts::Matrix Other>
  MatrixBase(const Other &other) : MatrixBase(other.expression()) {}

  auto operator=(concepts::Matrix auto const &v) -> MatrixBase & {
    return operator=(v.expression());
  }

  template <concepts::Expression Other>
  auto operator=(const Other &other) -> MatrixBase &
    requires(expression_traits::is_writable)
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
    decltype(auto) r = Base::operator()(std::forward<Args>(idxs)...);
    if constexpr (std::is_same_v<std::decay_t<decltype(r)>, value_type>) {
      return r;
    } else {
      using R = typename std::decay_t<decltype(r)>;
      if constexpr (R::extents_type::rank() == 1) {
        return VectorBase<R>(r);
      } else if constexpr (R::extents_type::rank() == 2) {
        return MatrixBase<R>(r);
      }
    }
  }
  template <typename... Args>
  auto operator()(Args &&...idxs) -> decltype(auto)

  {
    decltype(auto) r = Base::operator()(std::forward<Args>(idxs)...);
    if constexpr (std::is_same_v<std::decay_t<decltype(r)>, value_type>) {
      return r;
    } else {
      using R = typename std::decay_t<decltype(r)>;
      if constexpr (R::extents_type::rank() == 1) {
        return VectorBase<R>(r);
      } else if constexpr (R::extents_type::rank() == 2) {
        return MatrixBase<R>(r);
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
    auto v = Base::template slice_expression<Slices...>();
    using V = std::decay_t<decltype(v)>;
    if constexpr (V::extents_type::rank() == 2) {
      return MatrixBase<V>(std::move(v));
    } else {
      static_assert(V::extents_type::rank() == 1);
      return VectorBase<V>(std::move(v));
    }
  }

  template <typename... Slices> auto slice(Slices &&...slices) const {
    auto v =
        Base::template slice_expression<Slices...>(std::forward<Slices>(slices)...);
    using V = std::decay_t<decltype(v)>;
    if constexpr (V::extents_type::rank() == 2) {
      return MatrixBase<V>(std::move(v));
    } else {
      static_assert(V::extents_type::rank() == 1);
      return VectorBase<V>(std::move(v));
    }
  }
  template <typename... Slices> auto slice() const {
    auto v = Base::template slice_expression<Slices...>();
    using V = std::decay_t<decltype(v)>;
    if constexpr (V::extents_type::rank() == 2) {
      return MatrixBase<V>(std::move(v));
    } else {
      static_assert(V::extents_type::rank() == 1);
      return VectorBase<V>(std::move(v));
    }
  }

  template <typename... Slices> auto slice(Slices &&...slices) {
    auto v =
        Base::template slice_expression<Slices...>(std::forward<Slices>(slices)...);
    using V = std::decay_t<decltype(v)>;
    if constexpr (V::extents_type::rank() == 2) {
      return MatrixBase<V>(std::move(v));
    } else {
      static_assert(V::extents_type::rank() == 1);
      return VectorBase<V>(std::move(v));
    }
  }

  auto diagonal() const {
    return VectorBase<expression::unary::Diagonal<const expression_type>>(expression());
  }
  auto diagonal() {
    return VectorBase<expression::unary::Diagonal<expression_type>>(expression());
  }

  template <rank_type... ranks> auto swizzle() const {
    return Base::template swizzle<MatrixBase, ranks...>();
  }
  auto transpose() const { return Base::template swizzle<MatrixBase, 1, 0>(); }

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
    return detail::PartialReductionDispatcher<VectorBase, expression_type, 1>(expression());
  }
  auto colwise() {
    // we're reducing the first rows
    return detail::PartialReductionDispatcher<VectorBase, expression_type, 0>(expression());
  }
  auto rowwise() const {
    // we're reducing the first cols
    return detail::PartialReductionDispatcher<VectorBase, const expression_type, 1>(
        expression());
  }
  auto colwise() const {
    // we're reducing the first rows
    return detail::PartialReductionDispatcher<VectorBase, const expression_type, 0>(
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

UNARY_DECLARATION(MatrixBase, LogicalNot, operator!)
UNARY_DECLARATION(MatrixBase, BitNot, operator~)
UNARY_DECLARATION(MatrixBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(MatrixBase, Divides, operator/)

BINARY_DECLARATION(MatrixBase, Plus, operator+)
BINARY_DECLARATION(MatrixBase, Minus, operator-)
//

template <concepts::Matrix Expr1, concepts::Matrix Expr2>
auto operator==(Expr1 const &lhs, Expr2 const &rhs) {
  return (lhs.as_array() == rhs.as_array()).all();
}
template <concepts::Matrix Expr1, concepts::Matrix Expr2>
auto operator!=(Expr1 const &lhs, Expr2 const &rhs) {
  return (lhs.as_array() != rhs.as_array()).any();
}

template <concepts::Matrix Expr1, concepts::Matrix Expr2>
auto operator*(Expr1 const &lhs, Expr2 const &rhs) {
  using V =
      expression::binary::MatrixProduct<const typename Expr1::expression_type,
                                         const typename Expr2::expression_type>;
  return MatrixBase<V>(V(lhs.expression(), rhs.expression()));
}

template <concepts::Matrix Expr>
auto operator*(Expr const &lhs, typename Expr::value_type const &rhs) {
  using V = expression::unary::ScalarMultiplies<
      typename Expr::value_type, const typename Expr::expression_type, true>;
  return MatrixBase<V>(V(lhs.expression(), rhs));
}
template <concepts::Matrix Expr>
auto operator*(typename Expr::value_type const &lhs, Expr const &rhs) {
  using V = expression::unary::ScalarMultiplies<
      typename Expr::value_type, const typename Expr::expression_type, false>;
  return MatrixBase<V>(V(lhs, rhs.expression()));
}

template <concepts::Matrix Expr1, concepts::Vector Expr2>
auto operator*(Expr1 const &lhs, Expr2 const &rhs) {
  using V = expression::binary::MatrixVectorProduct<
      const typename Expr1::expression_type, const typename Expr2::expression_type>;

  return VectorBase<V>(V(lhs.expression(), rhs.expression()));
}

namespace concepts::detail {
template <typename T>
struct IsMatrix<MatrixBase<T>> : std::true_type {};
template <typename T>
struct IsZipperBase<MatrixBase<T>> : std::true_type {};
} // namespace concepts::detail

} // namespace zipper
#endif
