#if !defined(ZIPPER_MATRIX_HPP)
#define ZIPPER_MATRIX_HPP

#include "concepts/Matrix.hpp"
#include "expression/nullary/MDArray.hpp"
#include "storage/layout_types.hpp"
//
#include "MatrixBase.hpp"
#include "storage/SpanStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

// Default choice of RowMajor is defined in MatrixBase
template <typename ValueType, index_type Rows, index_type Cols, bool RowMajor>
class Matrix
    : public MatrixBase<
          expression::nullary::MDArray<ValueType, zipper::extents<Rows, Cols>,
                                       storage::matrix_layout<RowMajor>,
                                       default_accessor_policy<ValueType>>> {
public:
  using layout_type = storage::matrix_layout<RowMajor>;
  using expression_type =
      expression::nullary::MDArray<ValueType, zipper::extents<Rows, Cols>,
                                   storage::matrix_layout<RowMajor>,
                                   default_accessor_policy<ValueType>>;
  using Base = MatrixBase<expression_type>;

  using Base::expression;
  using value_type = Base::value_type;
  using extents_type = Base::extents_type;
  using Base::col;
  using Base::extent;
  using Base::extents;
  using Base::row;
  using extents_traits = detail::ExtentsTraits<extents_type>;
  using span_expression_type =
      expression::nullary::MDSpan<ValueType, zipper::extents<Rows, Cols>,
                                  storage::matrix_layout<RowMajor>,
                                  default_accessor_policy<ValueType>>;
  using const_span_expression_type =
      expression::nullary::MDSpan<const ValueType, zipper::extents<Rows, Cols>,
                                  storage::matrix_layout<RowMajor>,
                                  default_accessor_policy<ValueType>>;
  using span_type = MatrixBase<span_expression_type>;
  using const_span_type = MatrixBase<const_span_expression_type>;
  using Base::transpose;
  constexpr static bool is_static = extents_traits::is_static;

  Matrix()
      // requires(extents_traits::is_static)
      = default;

  template <typename T>
  Matrix(const std::initializer_list<std::initializer_list<T>> &l)
    requires(extents_traits::is_static)
  {
    assert(l.size() == extent(0));
    auto it = l.begin();
    for (index_type j = 0; j < l.size(); ++j, ++it) {
      assert(it->size() == extent(1));
      row(j) = *it;
    }
  }
  template <typename T>
  Matrix(const std::initializer_list<std::initializer_list<T>> &l)
    requires(extents_traits::rank_dynamic == 1)
      : Base(extents_type(extents_traits::is_dynamic_extent(0)
                              ? l.size()
                              : l.begin()->size())) {
    auto it = l.begin();
    for (index_type j = 0; j < l.size(); ++j, ++it) {
      assert(it->size() == extent(1));
      row(j) = *it;
    }
  }
  template <typename T>
  Matrix(const std::initializer_list<std::initializer_list<T>> &l)
    requires(extents_traits::rank_dynamic == 2)
      : Base(extents_type(l.size(), l.begin()->size())) {
    auto it = l.begin();
    for (index_type j = 0; j < l.size(); ++j, ++it) {
      assert(it->size() == extent(1));
      row(j) = *it;
    }
  }

  auto as_span() -> span_type {
    if constexpr (is_static) {
      return span_type(expression().as_std_span());
    } else {
      return span_type(expression().as_std_span(), extents());
    }
  }
  auto as_const_span() const -> const_span_type {
    if constexpr (is_static) {
      return const_span_type(expression().as_std_span());
    } else {
      return const_span_type(expression().as_std_span(), extents());
    }
  }
  auto as_span() const -> const_span_type { return as_const_span(); }

  Matrix(index_type dyn_size)
    requires(extents_traits::rank_dynamic == 1)
      : Base(extents_type(dyn_size)) {}

  Matrix(index_type rows, index_type cols)
    requires(extents_traits::is_dynamic)
      : Base(extents_type(rows, cols)) {}

  Matrix(const extents_type &e)
    requires(extents_traits::is_dynamic)
      : Base(e) {}
  Matrix(const extents_type &)
    requires(extents_traits::is_static)
      : Base() {}

#if defined(NDEBUG)
  Matrix(index_type, index_type)
#else
  Matrix(index_type rows, index_type cols)
#endif
    requires(extents_traits::is_static)
      : Base() {
    assert(rows == extent(0));
    assert(cols == extent(1));
  }

  template <concepts::Matrix Other> Matrix(const Other &other) : Base(other) {}

  template <concepts::Expression Other>
  Matrix(const Other &other) : Base(other) {}

  Matrix(const Matrix &other) : Base(other.expression()) {}

  template <index_type R2, index_type C2>
  Matrix(const Matrix<value_type, R2, C2> &other) : Base(other.expression()) {}

  using Base::operator=;
  auto operator=(Matrix &&other) -> Matrix & {
    Base::operator=(std::move(other.expression()));
    return *this;
  }

  auto operator=(const Matrix &other) -> Matrix & {
    Base::operator=(other.expression());
    return *this;
  }
  template <index_type R2, index_type C2>
  auto operator=(const Matrix<value_type, R2, C2> &other) -> Matrix & {
    Base::operator=(other.expression());
    return *this;
  }
};
template <concepts::Matrix MB>
Matrix(const MB &o) -> Matrix<std::decay_t<typename MB::value_type>,
                              MB::extents_type::static_extent(0),
                              MB::extents_type::static_extent(1)>;

namespace concepts::detail {
template <typename T, index_type R, index_type C, bool RowMajor>
struct IsMatrixBaseDerived<zipper::Matrix<T, R, C, RowMajor>> : std::true_type {
};

} // namespace concepts::detail
} // namespace zipper

#endif
