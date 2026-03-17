/// @file ZeroAwareOperation.hpp
/// @brief Coefficient-wise binary expression that propagates structural zero
///        information through addition and subtraction.
/// @ingroup expressions_binary sparsity
///
/// `ZeroAwareOperation<A, B, Op>` is functionally identical to `Operation<A,
/// B, Op>` for coefficient evaluation.  The difference is that it:
///   1. Sets `has_index_set = true` in its ExpressionTraits.
///   2. Provides `index_set<D>(other_indices...)` methods that return
///      the **union** of both children's non-zero index sets.
///
/// This enables downstream consumers (especially matrix products) to skip
/// computations in known-zero regions.
///
/// @note  The constrained `operator+`/`operator-` overloads in each
///        `*Base.hxx` file select `ZeroAwareOperation` over `Operation` when
///        at least one operand satisfies `HasIndexSet`.
///
/// @see zipper::expression::detail::IndexSet — concept satisfied by the
///      range types returned by index_set queries.
/// @see zipper::expression::detail::ContiguousIndexRange — the range type
///      returned by ZeroAwareOperation::index_set (conservative union).
/// @see zipper::expression::detail::HasIndexSet — compile-time trait that
///      triggers selection of ZeroAwareOperation over plain Operation.
/// @see zipper::expression::binary::Operation — the non-zero-aware
///      coefficient-wise binary expression (used when neither operand has
///      known zeros).
/// @see zipper::expression::unary::TriangularView — an expression with known
///      zeros whose addition/subtraction produces a ZeroAwareOperation.
/// @see zipper::expression::nullary::Identity — another expression with known
///      zeros.
/// @see zipper::expression::binary::MatrixProduct — uses index_set for
///      intersection semantics (multiplication), complementing the union
///      semantics used here for addition/subtraction.

#pragma once

#include "BinaryExpressionBase.hpp"
#include "detail/CoeffWiseTraits.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/expression/detail/IndexSet.hpp"

#include "zipper/detail/fmt.hpp"
#if defined(ZIPPER_FMT_OVERRIDES_DISABLED)
#include "zipper/utils/extents/as_array.hpp"
#endif
#include "zipper/utils/extents/extents_formatter.hpp"

namespace zipper::expression {

namespace binary {
template <zipper::concepts::QualifiedExpression A,
          zipper::concepts::QualifiedExpression B, typename Op>
class ZeroAwareOperation;
}  // namespace binary

// ─── ExpressionDetail specialization ─────────────────────────────────────────

template <zipper::concepts::QualifiedExpression A,
          zipper::concepts::QualifiedExpression B, typename Op>
struct detail::ExpressionDetail<binary::ZeroAwareOperation<A, B, Op>>
    : public binary::detail::DefaultBinaryExpressionDetail<A, B> {
    using _Base = binary::detail::DefaultBinaryExpressionDetail<A, B>;
    using ATraits = typename _Base::ATraits;
    using BTraits = typename _Base::BTraits;
    using ConvertExtentsUtil =
        binary::detail::coeffwise_extents_values<
            typename ATraits::extents_type,
            typename BTraits::extents_type>;
};

// ─── ExpressionTraits specialization ─────────────────────────────────────────

template <zipper::concepts::QualifiedExpression A,
          zipper::concepts::QualifiedExpression B, typename Op>
struct detail::ExpressionTraits<binary::ZeroAwareOperation<A, B, Op>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B> {
    using _Detail =
        detail::ExpressionDetail<binary::ZeroAwareOperation<A, B, Op>>;
    using extents_type =
        typename _Detail::ConvertExtentsUtil::merged_extents_type;
    using value_type = decltype(std::declval<Op>()(
        std::declval<typename _Detail::ATraits::value_type>(),
        std::declval<typename _Detail::BTraits::value_type>()));

    /// The result expression always tracks structural zeros.
    constexpr static bool has_index_set = true;

    /// Backward-compatible alias for has_index_set.
    constexpr static bool has_known_zeros = has_index_set;
};

// ─── ZeroAwareOperation class ────────────────────────────────────────────────

namespace binary {

// Namespace alias so we can refer to expression::detail:: types without
// ambiguity with binary::detail:: inside the class body.
namespace _za_nz = ::zipper::expression::detail;

/// @brief Coefficient-wise binary expression with structural zero propagation.
///
/// This is a drop-in replacement for `Operation<A, B, Op>` that additionally
/// exposes `nonzero_range<D>(other_indices...)` methods.  Coefficient
/// evaluation is identical to `Operation`.
///
/// Template parameters follow the same convention as `Operation`:
///   - A, B: child expression types (qualified — may be `const E&` or `E`)
///   - Op:   coefficient-wise operation functor (e.g. std::plus<double>)
template <zipper::concepts::QualifiedExpression A,
          zipper::concepts::QualifiedExpression B, typename Op>
class ZeroAwareOperation
    : public BinaryExpressionBase<ZeroAwareOperation<A, B, Op>, A, B> {
   public:
    using self_type = ZeroAwareOperation<A, B, Op>;
    using traits =
        ::zipper::expression::detail::ExpressionTraits<self_type>;
    using detail_type =
        ::zipper::expression::detail::ExpressionDetail<self_type>;
    using extents_type = typename traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    using Base = BinaryExpressionBase<self_type, A, B>;
    using value_type = typename traits::value_type;
    constexpr static bool is_static = extents_traits::is_static;

    using a_extents_type = typename detail_type::ATraits::extents_type;
    using b_extents_type = typename detail_type::BTraits::extents_type;

    using lhs_expression_type = std::remove_reference_t<A>;
    using rhs_expression_type = std::remove_reference_t<B>;

    /// Whether each child has known structural zeros.
    constexpr static bool lhs_has_index_set =
        _za_nz::HasIndexSet<lhs_expression_type>;
    constexpr static bool rhs_has_index_set =
        _za_nz::HasIndexSet<rhs_expression_type>;

    using Base::lhs;
    using Base::rhs;

    // ── Construction ─────────────────────────────────────────────────────

    template <typename U, typename V>
        requires std::constructible_from<typename Base::lhs_storage_type,
                                         U &&> &&
                 std::constructible_from<typename Base::rhs_storage_type,
                                         V &&>
    ZeroAwareOperation(U &&a, V &&b, const Op &op = {})
        : Base(std::forward<U>(a), std::forward<V>(b)), m_op(op) {
        if (!valid_input_extents(lhs().extents(), rhs().extents())) {
            throw std::runtime_error(fmt::format(
                "ZeroAwareOperation between {} and {} is invalid",
#if defined(ZIPPER_FMT_OVERRIDES_DISABLED)
                zipper::utils::extents::as_array(lhs().extents()),
                zipper::utils::extents::as_array(rhs().extents())));
#else
                lhs().extents(), rhs().extents()));
#endif
        }
    }

    // ── Coefficient evaluation (same as Operation) ───────────────────────

    value_type get_value(const auto &a, const auto &b) const {
        return m_op(a, b);
    }

    constexpr auto extent(rank_type i) const -> index_type {
        if constexpr (a_extents_type::rank() == 0) {
            return rhs().extent(i);
        } else {
            return lhs().extent(i);
        }
    }

    constexpr auto extents() const -> extents_type {
        return extents_traits::make_extents_from(*this);
    }

    // ── Index set interface ────────────────────────────────────────────
    //
    // For addition/subtraction, the result can be nonzero wherever EITHER
    // child is nonzero → the index set is the UNION of both sets.
    //
    // We provide per-dimension queries.  If a child doesn't have an
    // index set, we assume its entire extent is nonzero (FullRange).
    //
    // The union of two ranges is returned as:
    //   - If both are contiguous: a ContiguousIndexRange spanning the union
    //   - Otherwise: a SparseIndexRange with sorted merged indices
    //
    // For simplicity, we always return a ContiguousIndexRange covering
    // [min(first), max(last)) — this is a superset (conservative) union
    // that is cheap to compute.  A tighter union (SparseIndexRange) would
    // be more precise but requires allocation.

    /// @brief Type-preserving index set for dimension D.
    ///
    /// Returns the union of both children's index sets for the given
    /// dimension, preserving the exact range type (including DisjointRange)
    /// through `range_union()`.
    ///
    /// Rank-2 (matrices): D=0 → row range for a given column,
    ///                    D=1 → column range for a given row.
    template <rank_type D>
    auto index_set(index_type other_idx) const
        requires(extents_type::rank() == 2)
    {
        auto lhs_range = get_child_range<D>(lhs(), other_idx);
        auto rhs_range = get_child_range<D>(rhs(), other_idx);
        return _za_nz::range_union(lhs_range, rhs_range);
    }

    /// Rank-1 (vectors): nonzero segment (no arguments).
    template <rank_type D>
    auto index_set() const
        requires(D == 0 && extents_type::rank() == 1)
    {
        auto lhs_range = get_child_range_r1(lhs());
        auto rhs_range = get_child_range_r1(rhs());
        return _za_nz::range_union(lhs_range, rhs_range);
    }

    /// @deprecated Use index_set instead.
    template <rank_type D>
    auto nonzero_range(index_type other_idx) const
        requires(extents_type::rank() == 2)
    {
        return index_set<D>(other_idx);
    }

    /// @deprecated Use index_set instead.
    template <rank_type D>
    auto nonzero_range() const
        requires(D == 0 && extents_type::rank() == 1)
    {
        return index_set<D>();
    }

    // ── Convenience aliases (rank-2) ─────────────────────────────────────

    auto col_range_for_row(index_type row) const
        requires(extents_type::rank() == 2)
    {
        return index_set<1>(row);
    }

    auto row_range_for_col(index_type col) const
        requires(extents_type::rank() == 2)
    {
        return index_set<0>(col);
    }

    // ── Convenience alias (rank-1) ───────────────────────────────────────

    auto nonzero_segment() const
        requires(extents_type::rank() == 1)
    {
        return index_set<0>();
    }

    /// Deep copy.
    auto make_owned() const {
        auto owned_a = lhs().make_owned();
        auto owned_b = rhs().make_owned();
        return ZeroAwareOperation<decltype(owned_a), decltype(owned_b),
                                  Op>(std::move(owned_a),
                                      std::move(owned_b), m_op);
    }

   private:
    Op m_op;

    /// Validate that the two children have compatible extents.
    constexpr static bool valid_input_extents(const a_extents_type &a,
                                              const b_extents_type &b) {
        if constexpr (a_extents_type::rank() == 0 ||
                      b_extents_type::rank() == 0) {
            return true;
        } else if constexpr (a_extents_type::rank() ==
                             b_extents_type::rank()) {
            for (rank_type j = 0; j < a_extents_type::rank(); ++j) {
                if (a.extent(j) != b.extent(j) &&
                    a.extent(j) != std::dynamic_extent &&
                    b.extent(j) != std::dynamic_extent) {
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    // ── Helpers for extracting child ranges ──────────────────────────────

    /// Get the index set for dimension D from a rank-2 child.
    /// If the child doesn't have an index set, returns a FullRange covering
    /// the child's extent along dimension D.  Otherwise, returns the
    /// child's native range type (preserving DisjointRange, etc.).
    template <rank_type D, typename Child>
    static auto get_child_range(const Child &child, index_type other_idx) {
        if constexpr (_za_nz::HasIndexSet<Child>) {
            return child.template index_set<D>(other_idx);
        } else {
            // Dense child: assume all indices are nonzero.
            return _za_nz::ContiguousIndexRange{
                .first = 0, .last = child.extent(D)};
        }
    }

    /// Get the index set from a rank-1 child (no arguments).
    template <typename Child>
    static auto get_child_range_r1(const Child &child) {
        if constexpr (_za_nz::HasIndexSet<Child>) {
            return child.template index_set<0>();
        } else {
            return _za_nz::ContiguousIndexRange{
                .first = 0, .last = child.extent(0)};
        }
    }
};

// Deduction guide.
template <zipper::concepts::Expression A,
          zipper::concepts::Expression B, typename Op>
ZeroAwareOperation(const A &a, const B &b, const Op &op)
    -> ZeroAwareOperation<const A &, const B &, Op>;

}  // namespace binary

// ─── Zero-aware arithmetic aliases ───────────────────────────────────────────

namespace binary {

template <zipper::concepts::QualifiedExpression ExprA,
          zipper::concepts::QualifiedExpression ExprB>
using ZeroAwarePlus = ZeroAwareOperation<
    ExprA, ExprB,
    std::plus<typename ::zipper::expression::detail::ExpressionTraits<
        std::decay_t<ExprA>>::value_type>>;

template <zipper::concepts::QualifiedExpression ExprA,
          zipper::concepts::QualifiedExpression ExprB>
using ZeroAwareMinus = ZeroAwareOperation<
    ExprA, ExprB,
    std::minus<typename ::zipper::expression::detail::ExpressionTraits<
        std::decay_t<ExprA>>::value_type>>;

}  // namespace binary

}  // namespace zipper::expression
