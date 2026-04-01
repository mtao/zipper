#if !defined(ZIPPER_EXPRESSION_UNARY_TRIANGULARVIEW_HPP)
#define ZIPPER_EXPRESSION_UNARY_TRIANGULARVIEW_HPP

/// @file TriangularView.hpp
/// @brief A unary expression that presents a triangular view of a rank-2
///        (matrix) expression.
/// @ingroup expressions_unary sparsity
///
/// TriangularView<Mode, ExpressionType> wraps an existing matrix expression
/// and returns the child's coefficient for positions that lie within the
/// requested triangle, and zero for all positions outside it.  The diagonal
/// can be passed through as-is, replaced with ones (unit), or replaced
/// with zeros (strictly).
///
/// The triangular mode is a compile-time non-type template parameter (NTTP)
/// encoded as a bitmask enum, following the same naming conventions as
/// Eigen's triangular view modes.
///
/// Because CTAD cannot deduce NTTPs, a convenience factory function
/// `triangular_view<Mode>(expr)` is provided as the primary user API.
///
/// This expression is read-only: it does not expose `coeff_ref` or `assign`.
///
/// @see zipper::expression::unary::TriangularView::solve — forward/back
///      substitution solver method on TriangularView.
/// @see zipper::expression::detail::ContiguousIndexRange — the range type
///      returned by TriangularView::index_set.
/// @see zipper::expression::detail::IndexSet — concept satisfied by the
///      range types used in the zero-aware sparsity protocol.
/// @see zipper::expression::binary::MatrixProduct — uses index_set for
///      zero-aware dot products when multiplying triangular matrices.
/// @see zipper::expression::nullary::Identity — another expression with
///      known structural zeros (diagonal-only).
/// @see zipper::utils::decomposition::qr_solve — wraps the R factor in a
///      TriangularView for back-substitution.

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/concepts/Vector.hpp"
#include "zipper/expression/TriangularMode.hpp"
#include "zipper/expression/detail/IndexSet.hpp"
#include "zipper/utils/solver/detail/triangular_substitute.hpp"

namespace zipper::expression {

/// @brief Combined constraint for TriangularView: the mode must be valid
///        and the child expression must be rank-2.
template <TriangularMode Mode, typename ExpressionType>
concept ValidTriangularView =
    zipper::concepts::QualifiedExpression<ExpressionType> &&
    is_valid_triangular_mode<Mode>() &&
    (zipper::expression::detail::ExpressionTraits<
         std::decay_t<ExpressionType>>::extents_type::rank() == 2);

namespace unary {

/// @brief Forward declaration so that ExpressionTraits can be specialized
///        before the class body.
template <TriangularMode Mode,
          zipper::concepts::QualifiedExpression ExpressionType>
    requires ValidTriangularView<Mode, ExpressionType>
class TriangularView;

} // namespace unary

// ─────────────────────────────────────────────────────────────────────────────
// ExpressionTraits specialization for TriangularView.
//
// Key choices:
//   - is_value_based = false : the result depends on the position (row, col),
//     not just the child's scalar value, so the base class's default coeff()
//     delegation (which calls get_value) does not apply.
//   - is_coefficient_consistent = false : TriangularView zeros out entries,
//     so writing through it would not round-trip correctly.
//   - Extents are unchanged from the child expression.
//   - Access is read-only: is_const = true, is_reference = false.
// ─────────────────────────────────────────────────────────────────────────────
template <TriangularMode Mode,
          zipper::concepts::QualifiedExpression ExpressionType>
    requires ValidTriangularView<Mode, ExpressionType>
struct detail::ExpressionTraits<unary::TriangularView<Mode, ExpressionType>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          ExpressionType,
          zipper::detail::AccessFeatures{.is_const = true,
                                         .is_reference = false}> {
    using Base =
        zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
            ExpressionType,
            zipper::detail::AccessFeatures{.is_const = true,
                                           .is_reference = false}>;

    /// Override: position-dependent, not value-based.
    constexpr static bool is_value_based = false;

    /// Override: zeroes out entries, so not coefficient-consistent.
    constexpr static bool is_coefficient_consistent = false;

    /// TriangularView has structurally known zero regions.
    constexpr static bool has_index_set = true;

    /// Backward-compatible alias.
    constexpr static bool has_known_zeros = has_index_set;
};

namespace unary {

// ─────────────────────────────────────────────────────────────────────────────
/// @class TriangularView
/// @brief Presents a read-only triangular view of a rank-2 matrix expression.
///
/// @tparam Mode            Compile-time TriangularMode flag (NTTP).
/// @tparam ExpressionType  The child expression type (ref-qualified).
///
/// For a given (row, col) position, `coeff(row, col)` returns:
///   1. **Outside the triangle**: zero.
///   2. **On the diagonal**:
///      - If UnitDiag is set: 1.
///      - If ZeroDiag is set: 0.
///      - Otherwise: the child's value at (row, col).
///   3. **Inside the triangle** (not on diagonal): the child's value.
///
/// "Inside the triangle" means:
///   - For Lower (bit 0x1): row > col  (below diagonal).
///   - For Upper (bit 0x2): row < col  (above diagonal).
///
/// Usage (via factory function):
/// @code
///   auto M = Matrix33<double>({...});
///   auto L = triangular_view<TriangularMode::Lower>(M);
///   // L.coeff(2, 0) == M.coeff(2, 0)  (in lower triangle)
///   // L.coeff(0, 2) == 0.0             (outside lower triangle)
///   // L.coeff(1, 1) == M.coeff(1, 1)   (diagonal, pass-through)
/// @endcode
// ─────────────────────────────────────────────────────────────────────────────
template <TriangularMode Mode,
          zipper::concepts::QualifiedExpression ExpressionType>
    requires ValidTriangularView<Mode, ExpressionType>
class TriangularView
    : public UnaryExpressionBase<TriangularView<Mode, ExpressionType>,
                                 ExpressionType> {
public:
    // ── Type aliases ─────────────────────────────────────────────────────
    using self_type = TriangularView<Mode, ExpressionType>;
    using traits =
        zipper::expression::detail::ExpressionTraits<self_type>;
    using extents_type = typename traits::extents_type;
    using value_type = typename traits::value_type;
    using Base = UnaryExpressionBase<self_type, ExpressionType>;
    using Base::expression;

    using child_traits =
        zipper::expression::detail::ExpressionTraits<std::decay_t<ExpressionType>>;
    using child_extents_type = typename child_traits::extents_type;

    // ── Constructors ─────────────────────────────────────────────────────
    TriangularView(const TriangularView &o)
        : TriangularView(o.expression()) {}
    TriangularView(TriangularView &&o)
        : TriangularView(o.expression()) {}

    auto operator=(const TriangularView &) -> TriangularView & = delete;
    auto operator=(TriangularView &&) -> TriangularView & = delete;

    template <typename U>
        requires std::constructible_from<typename Base::storage_type, U &&>
    TriangularView(U &&b) : Base(std::forward<U>(b)) {}

    // ── make_owned ───────────────────────────────────────────────────────
    /// @brief Recursively deep-copy the child so the result owns all data.
    auto make_owned() const {
        auto owned_child = expression().make_owned();
        return TriangularView<Mode, const decltype(owned_child)>(
            std::move(owned_child));
    }

    // ── Compile-time mode ────────────────────────────────────────────────
    /// @brief The triangular mode used by this view (compile-time constant).
    constexpr static TriangularMode mode = Mode;

    // ── Coefficient access ───────────────────────────────────────────────
    /// @brief Returns the triangular-view coefficient at (row, col).
    ///
    /// This is the core logic of the expression:
    ///   - If (row, col) is on the diagonal: apply UnitDiag/ZeroDiag/pass-through.
    ///   - If (row, col) is inside the selected triangle: return child value.
    ///   - Otherwise: return zero.
    template <typename RowIdx, typename ColIdx>
    auto coeff(RowIdx &&row, ColIdx &&col) const -> value_type {
        const auto r = static_cast<index_type>(row);
        const auto c = static_cast<index_type>(col);

        // ── Diagonal ────────────────────────────────────────────────
        if (r == c) {
            if constexpr (has_flag(Mode, TriangularMode::UnitDiag)) {
                return value_type{1};
            } else if constexpr (has_flag(Mode, TriangularMode::ZeroDiag)) {
                return value_type{0};
            } else {
                // Pass through the child's diagonal value.
                return expression().coeff(
                    std::forward<RowIdx>(row), std::forward<ColIdx>(col));
            }
        }

        // ── Inside the triangle ─────────────────────────────────────
        if constexpr (has_flag(Mode, TriangularMode::Lower) &&
                      has_flag(Mode, TriangularMode::Upper)) {
            // OffDiagonal: everything except diagonal (already handled above).
            return expression().coeff(
                std::forward<RowIdx>(row), std::forward<ColIdx>(col));
        } else if constexpr (has_flag(Mode, TriangularMode::Lower)) {
            // Lower triangle: row > col
            if (r > c) {
                return expression().coeff(
                    std::forward<RowIdx>(row), std::forward<ColIdx>(col));
            }
        } else {
            // Upper triangle: row < col
            if (r < c) {
                return expression().coeff(
                    std::forward<RowIdx>(row), std::forward<ColIdx>(col));
            }
        }

        // ── Outside the triangle ────────────────────────────────────
        return value_type{0};
    }

    // ── Index set queries ─────────────────────────────────────────────
    // These methods report the structurally non-zero index range along
    // dimension D, given the index in the other dimension.
    //
    // For rank-2 expressions:
    //   index_set<1>(row) = column range with non-zeros in that row
    //   index_set<0>(col) = row range with non-zeros in that column

    /// @brief Returns the non-zero index range along dimension @p D.
    ///
    /// For D==1 (column range given a row):
    ///   Lower:          [0, row+1)
    ///   StrictlyLower:  [0, row)
    ///   UnitLower:      [0, row+1)
    ///   Upper:          [row, ncols)
    ///   StrictlyUpper:  [row+1, ncols)
    ///   UnitUpper:      [row, ncols)
    ///   OffDiagonal:    [0, row) ∪ [row+1, ncols)
    ///
    /// For D==0 (row range given a column):
    ///   Lower:          [col, nrows)
    ///   StrictlyLower:  [col+1, nrows)
    ///   UnitLower:      [col, nrows)
    ///   Upper:          [0, col+1)
    ///   StrictlyUpper:  [0, col)
    ///   UnitUpper:      [0, col+1)
    ///   OffDiagonal:    [0, col) ∪ [col+1, nrows)
    template <rank_type D>
        requires(D < 2)
    auto index_set(index_type other_idx) const {
        using CR = zipper::expression::detail::ContiguousIndexRange;
        // Contiguous range starting at compile-time 0 (zero storage for
        // the lower bound thanks to [[no_unique_address]]).
        using CR0 = zipper::expression::detail::ContiguousIndexSet<
            static_index_t<0>, index_type>;

        if constexpr (D == 1) {
            // Column range for a given row
            const auto ncols = Base::extent(1);

            if constexpr (has_flag(Mode, TriangularMode::Lower) &&
                          has_flag(Mode, TriangularMode::Upper)) {
                // OffDiagonal: [0, row) ∪ [row+1, ncols)
                return zipper::expression::detail::DisjointRange<CR0, CR>{
                    std::tuple{
                        CR0{static_index_t<0>{}, other_idx},
                        CR{std::min(other_idx + 1, ncols), ncols}}};
            } else if constexpr (has_flag(Mode, TriangularMode::Lower)) {
                if constexpr (has_flag(Mode, TriangularMode::ZeroDiag)) {
                    // StrictlyLower: cols [0, row)
                    return CR0{static_index_t<0>{}, other_idx};
                } else {
                    // Lower or UnitLower: cols [0, row+1)
                    return CR0{static_index_t<0>{},
                               std::min(other_idx + 1, ncols)};
                }
            } else {
                if constexpr (has_flag(Mode, TriangularMode::ZeroDiag)) {
                    // StrictlyUpper: cols [row+1, ncols)
                    return CR{std::min(other_idx + 1, ncols), ncols};
                } else {
                    // Upper or UnitUpper: cols [row, ncols)
                    return CR{other_idx, ncols};
                }
            }
        } else {
            // Row range for a given column
            const auto nrows = Base::extent(0);

            if constexpr (has_flag(Mode, TriangularMode::Lower) &&
                          has_flag(Mode, TriangularMode::Upper)) {
                // OffDiagonal: [0, col) ∪ [col+1, nrows)
                return zipper::expression::detail::DisjointRange<CR0, CR>{
                    std::tuple{
                        CR0{static_index_t<0>{}, other_idx},
                        CR{std::min(other_idx + 1, nrows), nrows}}};
            } else if constexpr (has_flag(Mode, TriangularMode::Lower)) {
                if constexpr (has_flag(Mode, TriangularMode::ZeroDiag)) {
                    // StrictlyLower: rows [col+1, nrows)
                    return CR{std::min(other_idx + 1, nrows), nrows};
                } else {
                    // Lower or UnitLower: rows [col, nrows)
                    return CR{other_idx, nrows};
                }
            } else {
                if constexpr (has_flag(Mode, TriangularMode::ZeroDiag)) {
                    // StrictlyUpper: rows [0, col)
                    return CR0{static_index_t<0>{}, other_idx};
                } else {
                    // Upper or UnitUpper: rows [0, col+1)
                    return CR0{static_index_t<0>{},
                               std::min(other_idx + 1, nrows)};
                }
            }
        }
    }

    /// @brief Backward-compatible wrapper; delegates to index_set.
    template <rank_type D>
        requires(D < 2)
    auto nonzero_range(index_type other_idx) const {
        return index_set<D>(other_idx);
    }

    // ── Convenience aliases (rank-2) ─────────────────────────────────

    /// @brief Returns the non-zero column range for a given row.
    auto col_range_for_row(index_type row) const {
        return index_set<1>(row);
    }

    /// @brief Returns the non-zero row range for a given column.
    auto row_range_for_col(index_type col) const {
        return index_set<0>(col);
    }

    // ── Solve ────────────────────────────────────────────────────────────
    /// @brief Solve T * x = b using forward or back substitution.
    ///
    /// This dispatches to forward substitution (Lower modes) or back
    /// substitution (Upper modes) at compile time.
    ///
    /// @param  b  Right-hand side vector.
    /// @return `std::expected<Vector<T, Dim>, SolverError>` — the solution
    ///         on success, or a breakdown error on zero pivot.
    template <zipper::concepts::Vector BDerived>
        requires(!(has_flag(Mode, TriangularMode::Lower) &&
                   has_flag(Mode, TriangularMode::Upper)))
    auto solve(const BDerived &b) const {
        using T = typename BDerived::value_type;
        constexpr auto Dim = BDerived::extents_type::static_extent(0);

        using ResultVec = zipper::Vector<T, Dim>;
        using Result =
            std::expected<ResultVec, zipper::utils::solver::SolverError>;

        ResultVec x(b);

        std::expected<void, zipper::utils::solver::SolverError> status;
        if constexpr (has_flag(Mode, TriangularMode::Lower)) {
            status = zipper::utils::solver::detail::
                forward_substitute_inplace<Mode>(*this, x);
        } else {
            status = zipper::utils::solver::detail::
                back_substitute_inplace<Mode>(*this, x);
        }

        if (!status) {
            return Result{std::unexpected(std::move(status.error()))};
        }
        return Result{std::move(x)};
    }
};

} // namespace unary

// ─────────────────────────────────────────────────────────────────────────────
/// @brief Factory function for creating TriangularView expressions.
///
/// Since CTAD cannot deduce non-type template parameters (NTTPs), this
/// function provides a clean API.
///
/// Two overloads are provided:
///   1. For raw expressions (types satisfying QualifiedExpression directly).
///   2. For ZipperBase-derived wrappers (e.g. Matrix, MatrixBase) that
///      expose an underlying expression via `.expression()`.
///
/// @code
///   auto M = Matrix<double, 3, 3>({...});
///   auto L = triangular_view<TriangularMode::Lower>(M);
///   auto U = triangular_view<TriangularMode::UnitUpper>(M.expression());
/// @endcode
///
/// @tparam Mode  The triangular mode (compile-time NTTP).
/// @param  expr  The matrix expression (or wrapper) to view.
/// @return A TriangularView expression wrapping the underlying expression
///         by const reference.
// ─────────────────────────────────────────────────────────────────────────────

/// Overload 1: accepts a raw expression directly.
template <TriangularMode Mode,
          zipper::concepts::QualifiedExpression ExpressionType>
auto triangular_view(const ExpressionType &expr) {
    return unary::TriangularView<Mode, const ExpressionType &>(expr);
}

/// Overload 2: accepts a ZipperBase-derived wrapper (Matrix, MatrixBase, etc.)
/// and extracts the underlying expression via `.expression()`.
template <TriangularMode Mode, typename ZipperType>
    requires(!zipper::concepts::QualifiedExpression<ZipperType> &&
             zipper::concepts::QualifiedExpression<
                 decltype(std::declval<const ZipperType &>().expression())>)
auto triangular_view(const ZipperType &wrapper) {
    using expr_type =
        std::remove_reference_t<decltype(wrapper.expression())>;
    return unary::TriangularView<Mode, const expr_type &>(
        wrapper.expression());
}

} // namespace zipper::expression

#endif
