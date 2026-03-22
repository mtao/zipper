#if !defined(ZIPPER_CSMATRIX_HPP)
#define ZIPPER_CSMATRIX_HPP

/// @file CSMatrix.hpp
/// @brief Sparse matrix in compressed format, parameterized by layout.
/// @ingroup user_types
///
/// `CSMatrix<T, Rows, Cols, LayoutPolicy>` is the unified user-facing
/// sparse matrix type:
///   - `CSMatrix<T, R, C, layout_right>` = CSR (compressed sparse row)
///   - `CSMatrix<T, R, C, layout_left>`  = CSC (compressed sparse column)
///
/// Construction:
///   - From a `COOMatrix`: `CSMatrix A(coo);` or `auto A = coo.to_csr();`
///   - From entries: `CSMatrix<double, 3, 3> A(entries);`
///   - From a `SparseCompressedAccessor`: move existing data.
///
/// Mutation:
///   - `coeff_ref(row, col)` returns a reference to an *existing* entry.
///     Throws for missing entries — use COO for structural changes.
///
/// Layout conversion:
///   - `as_csr()` returns a CSR copy (no-op if already CSR).
///   - `as_csc()` returns a CSC copy (no-op if already CSC).
///   - `to_coo()` returns a `COOMatrix` with the same data.

#include "SparseEntry.hpp"
#include "COOMatrix.hpp"
#include "concepts/Matrix.hpp"
#include "MatrixBase.hxx"
#include "storage/SparseCompressedAccessor.hpp"

namespace zipper {

// Forward declarations for as_csr()/as_csc() return types
template <typename ValueType, index_type Rows, index_type Cols,
          typename LayoutPolicy>
class CSMatrix;

template <typename ValueType, index_type Rows, index_type Cols,
          typename LayoutPolicy = storage::default_layout_policy>
class CSMatrix
    : public MatrixBase<
          storage::SparseCompressedAccessor<ValueType,
                                           zipper::extents<Rows, Cols>,
                                           LayoutPolicy>> {
public:
  using expression_type =
      storage::SparseCompressedAccessor<ValueType,
                                       zipper::extents<Rows, Cols>,
                                       LayoutPolicy>;
  using coo_expression_type =
      storage::SparseCoordinateAccessor<ValueType,
                                       zipper::extents<Rows, Cols>>;
  using Base = MatrixBase<expression_type>;
  using value_type = typename Base::value_type;
  using extents_type = typename Base::extents_type;
  using layout_policy = LayoutPolicy;
  using extents_traits = detail::ExtentsTraits<extents_type>;
  using Base::expression;
  using Base::extent;
  using Base::extents;
  constexpr static bool is_static = extents_traits::is_static;
  constexpr static bool is_csr =
      std::is_same_v<LayoutPolicy, storage::layout_right>;
  constexpr static bool is_csc =
      std::is_same_v<LayoutPolicy, storage::layout_left>;

  CSMatrix() = default;
  CSMatrix(const CSMatrix &) = default;
  CSMatrix(CSMatrix &&) = default;
  auto operator=(const CSMatrix &) -> CSMatrix & = default;
  auto operator=(CSMatrix &&) -> CSMatrix & = default;

  /// Assign from any expression with compatible extents.
  template <concepts::Expression Other>
  auto operator=(const Other &other) -> CSMatrix &
    requires(zipper::utils::extents::assignable_extents_v<
                 typename Other::extents_type, extents_type>)
  {
    expression().assign(other);
    return *this;
  }

  /// Assign from a Zipper-wrapped expression.
  template <concepts::Zipper Other>
  auto operator=(const Other &other) -> CSMatrix &
    requires(zipper::utils::extents::assignable_extents_v<
                 typename std::decay_t<Other>::extents_type, extents_type>)
  {
    expression().assign(other.expression());
    return *this;
  }

  /// Construct from any expression with compatible extents.
  template <concepts::Expression Other>
  CSMatrix(const Other &other)
    requires(zipper::utils::extents::assignable_extents_v<
                 typename Other::extents_type, extents_type>)
  {
    if constexpr (!is_static) {
      static_cast<extents_type &>(expression()) =
          extents_traits::convert_from(other.extents());
    }
    expression().assign(other);
  }

  /// Construct from a Zipper-wrapped expression.
  template <concepts::Zipper Other>
  CSMatrix(const Other &other)
    requires(!std::same_as<std::decay_t<Other>, CSMatrix> &&
             zipper::utils::extents::assignable_extents_v<
                 typename std::decay_t<Other>::extents_type, extents_type>)
  {
    if constexpr (!is_static) {
      static_cast<extents_type &>(expression()) =
          extents_traits::convert_from(other.extents());
    }
    expression().assign(other.expression());
  }

  // From compressed accessor
  CSMatrix(expression_type &&expr) : Base(std::move(expr)) {}
  CSMatrix(const expression_type &expr) : Base(expr) {}

  // From COOMatrix (compresses internally with this layout)
  CSMatrix(const COOMatrix<value_type, Rows, Cols> &coo)
      : Base(expression_type(coo.expression())) {}

  // From COO expression
  CSMatrix(const coo_expression_type &coo) : Base(expression_type(coo)) {}

  // Construct from range of SparseEntry<T, 2> (via COO intermediary)
  template <std::ranges::input_range R>
    requires std::same_as<std::ranges::range_value_t<R>,
                          SparseEntry<value_type, 2>>
  CSMatrix(const R &entries)
    requires(is_static)
      : CSMatrix(COOMatrix<value_type, Rows, Cols>(entries)) {}

  template <std::ranges::input_range R>
    requires std::same_as<std::ranges::range_value_t<R>,
                          SparseEntry<value_type, 2>>
  CSMatrix(const R &entries, const extents_type &e)
    requires(!is_static)
      : CSMatrix(COOMatrix<value_type, Rows, Cols>(entries, e)) {}

  // ── Element access ────────────────────────────────────────────────────
  auto coeff_ref(index_type row, index_type col) -> value_type & {
    return expression().coeff_ref(row, col);
  }

  auto const_coeff_ref(index_type row, index_type col) const
      -> const value_type & {
    return expression().const_coeff_ref(row, col);
  }

  auto compressed_data() const -> const auto & {
    return expression().compressed_data();
  }

  // ── Conversion to COO ─────────────────────────────────────────────────
  auto to_coo() const -> COOMatrix<value_type, Rows, Cols> {
    COOMatrix<value_type, Rows, Cols> result;
    if constexpr (!is_static) {
      result = COOMatrix<value_type, Rows, Cols>(extents());
    }
    const auto &cd = compressed_data();
    using Base0 =
        storage::detail::SparseCompressedData<
            std::remove_const_t<value_type>, 0>;
    const auto &base = static_cast<const Base0 &>(cd);
    for (const auto &[outer, start, sz] : cd.m_spans) {
      for (size_t i = start; i < start + sz; ++i) {
        auto inner = base.m_data[i].first;
        auto val = base.m_data[i].second;
        if constexpr (is_csr) {
          // outer = row, inner = col
          result.emplace(outer, inner) = val;
        } else {
          // CSC: outer = col, inner = row
          result.emplace(inner, outer) = val;
        }
      }
    }
    result.compress();
    return result;
  }

  // ── Layout conversion ─────────────────────────────────────────────────

  /// Convert to CSR format. No-op copy if already CSR, O(nnz) conversion
  /// through COO intermediary if CSC.
  auto as_csr() const -> CSMatrix<value_type, Rows, Cols, storage::layout_right> {
    if constexpr (is_csr) {
      return *this;
    } else {
      return CSMatrix<value_type, Rows, Cols, storage::layout_right>(to_coo());
    }
  }

  /// Convert to CSC format. No-op copy if already CSC, O(nnz) conversion
  /// through COO intermediary if CSR.
  auto as_csc() const -> CSMatrix<value_type, Rows, Cols, storage::layout_left> {
    if constexpr (is_csc) {
      return *this;
    } else {
      return CSMatrix<value_type, Rows, Cols, storage::layout_left>(to_coo());
    }
  }
};

// CTAD
template <typename VT, typename E, typename LP>
CSMatrix(storage::SparseCompressedAccessor<VT, E, LP>)
    -> CSMatrix<VT, E::static_extent(0), E::static_extent(1), LP>;

template <typename VT, index_type R, index_type C>
CSMatrix(const COOMatrix<VT, R, C> &)
    -> CSMatrix<VT, R, C, storage::default_layout_policy>;

namespace concepts::detail {
template <typename T, index_type R, index_type C, typename LP>
struct IsMatrix<zipper::CSMatrix<T, R, C, LP>> : std::true_type {};
template <typename T, index_type R, index_type C, typename LP>
struct IsZipperBase<zipper::CSMatrix<T, R, C, LP>> : std::true_type {};
} // namespace concepts::detail

// ── COOMatrix::to_csc() definition (needs CSMatrix to be complete) ─────
template <typename ValueType, index_type Rows, index_type Cols>
auto COOMatrix<ValueType, Rows, Cols>::to_csc() const
    -> CSMatrix<ValueType, Rows, Cols, storage::layout_left> {
  return CSMatrix<ValueType, Rows, Cols, storage::layout_left>(*this);
}

} // namespace zipper

#endif
