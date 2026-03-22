#if !defined(ZIPPER_EXPRESSION_DETAIL_SPARSEASSIGNHELPER_HPP)
#define ZIPPER_EXPRESSION_DETAIL_SPARSEASSIGNHELPER_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"
#include <tuple>

namespace zipper::expression::detail {

/// SparseAssignHelper populates a sparse target from an arbitrary source
/// expression.  Unlike AssignHelper (which writes every index via coeff_ref),
/// this helper clears the target and emplaces only nonzero values.
///
/// When the source has `has_index_set` (structurally known zeros), only the
/// nonzero indices are visited — giving an O(nnz) assignment.  Otherwise,
/// all indices are visited and zero values are skipped.
///
/// The Target type must provide:
///   - clear()                           — remove all stored entries
///   - emplace(indices...) -> value_type& — insert a new entry
///   - compress()                        — sort/deduplicate after bulk insert
///   - extents() -> extents_type         — shape of the sparse storage
///   - reserve(n) (optional)             — pre-allocate capacity
struct SparseAssignHelper {

  /// Assign from any expression into a rank-1 sparse target (COO vector).
  template <zipper::concepts::Expression From, typename To>
    requires(To::rank() == 1)
  static void assign(const From &from, To &to) {
    using FromTraits = ExpressionTraits<From>;

    to.clear();

    if constexpr (FromTraits::has_index_set) {
      // Source has structural sparsity — iterate only nonzero indices.
      auto idx_set = from.template index_set<0>();
      to.reserve(idx_set.size());
      for (auto j : idx_set) {
        auto val = from.coeff(j);
        if (val != typename From::value_type{0}) {
          to.emplace(j) = static_cast<typename To::value_type>(val);
        }
      }
    } else {
      // Dense source — iterate all indices, skip zeros.
      const auto ext = to.extents();
      for (index_type j = 0; j < ext.extent(0); ++j) {
        auto val = from.coeff(j);
        if (val != typename From::value_type{0}) {
          to.emplace(j) = static_cast<typename To::value_type>(val);
        }
      }
    }

    to.compress();
  }

  /// Assign from any expression into a rank-2 sparse target (COO matrix).
  ///
  /// When the source has `has_index_set`, detects `preferred_layout` to
  /// choose the optimal iteration order:
  ///   - CSC (layout_left) sources: iterate column-first
  ///   - CSR (layout_right) or no preference: iterate row-first
  template <zipper::concepts::Expression From, typename To>
    requires(To::rank() == 2)
  static void assign(const From &from, To &to) {
    using FromTraits = ExpressionTraits<From>;

    to.clear();

    if constexpr (FromTraits::has_index_set) {
      using pref = typename FromTraits::preferred_layout;

      // Detect CSC-preferred sources: iterate column-first for efficiency.
      constexpr bool is_csc_source = [] {
        if constexpr (zipper::detail::is_sparse_layout_preference_v<pref>) {
          return std::is_same_v<typename pref::layout_policy,
                                zipper::storage::layout_left>;
        } else {
          return false;
        }
      }();

      if constexpr (is_csc_source) {
        // Column-first iteration: for each col, get the row index set.
        const auto ext = to.extents();
        for (index_type j = 0; j < ext.extent(1); ++j) {
          auto row_set = from.template index_set<0>(j);
          for (auto i : row_set) {
            auto val = from.coeff(i, j);
            if (val != typename From::value_type{0}) {
              to.emplace(i, j) = static_cast<typename To::value_type>(val);
            }
          }
        }
      } else {
        // Row-first iteration (CSR or no preference).
        const auto ext = to.extents();
        for (index_type i = 0; i < ext.extent(0); ++i) {
          auto col_set = from.template index_set<1>(i);
          for (auto j : col_set) {
            auto val = from.coeff(i, j);
            if (val != typename From::value_type{0}) {
              to.emplace(i, j) = static_cast<typename To::value_type>(val);
            }
          }
        }
      }
    } else {
      // Dense source — iterate all indices, skip zeros.
      const auto ext = to.extents();
      for (index_type i = 0; i < ext.extent(0); ++i) {
        for (index_type j = 0; j < ext.extent(1); ++j) {
          auto val = from.coeff(i, j);
          if (val != typename From::value_type{0}) {
            to.emplace(i, j) = static_cast<typename To::value_type>(val);
          }
        }
      }
    }

    to.compress();
  }
};

} // namespace zipper::expression::detail
#endif
