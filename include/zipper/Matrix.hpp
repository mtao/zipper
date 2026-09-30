#if !defined(ZIPPER_MATRIX_HPP)
#define ZIPPER_MATRIX_HPP

/// @file Matrix.hpp
/// @brief Owning dense matrix type with static or dynamic extents.
/// @ingroup user_types
///
/// `Matrix<T, Rows, Cols, RowMajor>` is the primary user-facing matrix type.
/// It owns its data via an `MDArray` storage backend and inherits the full
/// `MatrixBase` interface (arithmetic operators, row/column access, transpose,
/// determinant, etc.).
///
/// Template parameters:
///   - `T`: scalar type (e.g. `double`, `float`).
///   - `Rows`: number of rows (`dynamic_extent` for runtime-sized).
///   - `Cols`: number of columns (`dynamic_extent` for runtime-sized).
///   - `RowMajor`: storage order (default: `true` = row-major).
///
/// Construction:
///   - Default: `Matrix<double, 3, 3> A;` (static, zero-initialised).
///   - From initializer lists: `Matrix<double, 3, 3>
///   A({{1,2,3},{4,5,6},{7,8,9}});`
///   - Dynamic: `Matrix<double, dynamic_extent, dynamic_extent> A(m, n);`
///     (zero-initialised)
///   - Uninitialized (opt-in, skips the zero-fill; write before reading):
///     `Matrix<double, 3, 3> A(zipper::uninitialized);`,
///     `MatrixXX<double> A(zipper::uninitialized, m, n);`
///   - Copy from expression: `Matrix<double, 3, 3> A(some_expression);`
///
/// @code
///   // Static 3x3 matrix
///   Matrix<double, 3, 3> A({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}});
///
///   // Dynamic matrix
///   Matrix<double, dynamic_extent, dynamic_extent> B(4, 4);
///
///   // Matrix arithmetic
///   auto C = A * A;        // matrix-matrix product
///   auto v = A * x;        // matrix-vector product
///   auto At = A.transpose();
///   double d = A.determinant();
///
///   // Row/column access
///   auto r0 = A.row(0);    // first row (FormBase)
///   auto c1 = A.col(1);    // second column (VectorBase)
///
///   // Span views (non-owning)
///   auto s = A.as_span();  // mutable span
///   auto cs = A.as_const_span();
/// @endcode
///
/// @see zipper::MatrixBase — CRTP base providing the matrix interface.
/// @see zipper::Vector — owning column vector type.
/// @see zipper::Form — owning row vector (1-form) type.
/// @see zipper::expression::nullary::MDArray — the underlying owning storage.
/// @see zipper::expression::unary::TriangularView — view a Matrix as
///      triangular; call `.solve(b)` for forward/back substitution.
/// @see zipper::utils::solver — iterative solvers that operate on matrices.
/// @see zipper::utils::decomposition::qr — QR factorisation of matrices.
/// @see zipper::utils::inverse — compute the matrix inverse.

#include "concepts/Matrix.hpp"
#include "detail/extents_check.hpp"
#include "expression/nullary/MDArray.hpp"
#include "storage/layout_types.hpp"
//
#include "MatrixBase.hxx"
// storage/SpanStorage.hpp has been removed; MDSpan is used for span types now
#include "expression/nullary/MDSpan.hpp"
#include "zipper/types.hpp"
namespace zipper {

// Default choice of RowMajor is defined in MatrixBase
template <typename ValueType, index_type Rows, index_type Cols, bool RowMajor>
class Matrix
  : public MatrixBase<
        expression::nullary::MDArray<ValueType,
                                     zipper::extents<Rows, Cols>,
                                     storage::matrix_layout<RowMajor>,
                                     default_accessor_policy<ValueType>>> {
  public:
    using layout_type = storage::matrix_layout<RowMajor>;
    using expression_type =
        expression::nullary::MDArray<ValueType,
                                     zipper::extents<Rows, Cols>,
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
        expression::nullary::MDSpan<ValueType,
                                    zipper::extents<Rows, Cols>,
                                    storage::matrix_layout<RowMajor>,
                                    default_accessor_policy<ValueType>>;
    using const_span_expression_type =
        expression::nullary::MDSpan<const ValueType,
                                    zipper::extents<Rows, Cols>,
                                    storage::matrix_layout<RowMajor>,
                                    default_accessor_policy<ValueType>>;
    using span_type = MatrixBase<span_expression_type>;
    using const_span_type = MatrixBase<const_span_expression_type>;
    using Base::transpose;
    constexpr static bool is_static = extents_traits::is_static;

    Matrix()
        // requires(extents_traits::is_static)
        = default;

    /// Throws std::invalid_argument on ragged rows or a static-extent mismatch.
    template <typename T>
    Matrix(const std::initializer_list<std::initializer_list<T>> &l)
      : Base(initializer_list_extents(l)) {
        auto it = l.begin();
        for (index_type j = 0; j < l.size(); ++j, ++it) { row(j) = *it; }
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

    /// Allocates storage without initializing the elements; every element
    /// must be written before it is read. Accepts either all extents or only
    /// the dynamic ones (none for fully static types). See
    /// zipper::uninitialized_t.
    template <typename... Args>
        requires((std::is_convertible_v<Args, index_type> && ...)
                 && (sizeof...(Args) == extents_type::rank()
                     || sizeof...(Args) == extents_type::rank_dynamic())
                 && std::is_constructible_v<typename Base::expression_type,
                                            uninitialized_t,
                                            const extents_type &>)
    explicit Matrix(uninitialized_t, Args &&...args)
      : Base(uninitialized, extents_type(static_cast<index_type>(args)...)) {
        if constexpr (sizeof...(Args) == extents_type::rank()
                      && sizeof...(Args) > 0) {
            detail::check_extents<extents_type>(
                static_cast<index_type>(args)...);
        }
    }
    Matrix(uninitialized_t, const extents_type &e)
        requires(std::is_constructible_v<typename Base::expression_type,
                                         uninitialized_t,
                                         const extents_type &>)
      : Base(uninitialized, e) {}

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

    Matrix([[maybe_unused]] index_type rows, [[maybe_unused]] index_type cols)
        requires(extents_traits::is_static)
      : Base() {
        detail::check_extents<extents_type>(rows, cols);
    }

    template <concepts::Matrix Other>
    Matrix(const Other &other) : Base(other) {}

    template <concepts::Expression Other>
    Matrix(const Other &other) : Base(other) {}

    Matrix(const Matrix &other) = default;

    template <index_type R2, index_type C2>
    Matrix(const Matrix<value_type, R2, C2> &other)
      : Base(other.expression()) {}

    using Base::operator=;
    auto operator=(Matrix &&other) -> Matrix & = default;
    auto operator=(const Matrix &other) -> Matrix & = default;
    template <index_type R2, index_type C2>
    auto operator=(const Matrix<value_type, R2, C2> &other) -> Matrix & {
        Base::operator=(other.expression());
        return *this;
    }

  private:
    template <typename T>
    static auto initializer_list_extents(
        const std::initializer_list<std::initializer_list<T>> &l)
        -> extents_type {
        const index_type rows = l.size();
        const index_type cols = rows > 0 ? l.begin()->size()
                                : extents_traits::is_dynamic_extent(1)
                                    ? index_type{0}
                                    : extents_type::static_extent(1);
        index_type j = 0;
        for (const auto &r : l) {
            if (r.size() != cols) {
                throw std::invalid_argument(
                    std::format("Matrix initializer_list: row {} has size {} "
                                "but expected {}",
                                j,
                                r.size(),
                                cols));
            }
            ++j;
        }
        detail::check_extents<extents_type>(rows, cols);
        return extents_type(rows, cols);
    }
};
template <concepts::MatrixExpression E>
Matrix(const E &) -> Matrix<std::decay_t<typename E::value_type>,
                            E::extents_type::static_extent(0),
                            E::extents_type::static_extent(1)>;

template <concepts::Matrix MB>
Matrix(const MB &o) -> Matrix<std::decay_t<typename MB::value_type>,
                              MB::extents_type::static_extent(0),
                              MB::extents_type::static_extent(1)>;

namespace concepts::detail {
    template <typename T, index_type R, index_type C, bool RowMajor>
    struct IsMatrix<zipper::Matrix<T, R, C, RowMajor>> : std::true_type {};
    template <typename T, index_type R, index_type C, bool RowMajor>
    struct IsZipperBase<zipper::Matrix<T, R, C, RowMajor>> : std::true_type {};

} // namespace concepts::detail
} // namespace zipper

#endif
