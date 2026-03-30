#if !defined(ZIPPER_EXPRESSION_DETAIL_EXPRESSIONTRAITS_HPP)
#define ZIPPER_EXPRESSION_DETAIL_EXPRESSIONTRAITS_HPP
#include "zipper/concepts/Extents.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/detail/Features.hpp"
#include "zipper/detail/LayoutPreference.hpp"
#include "zipper/types.hpp"

namespace zipper::expression::detail {

using AccessFeatures = zipper::detail::AccessFeatures;
using ShapeFeatures = zipper::detail::ShapeFeatures;
template <typename T,
          zipper::concepts::Extents Extents,
          AccessFeatures AF = AccessFeatures::from_type<T>(),
          ShapeFeatures SF = ShapeFeatures{}>

struct BasicExpressionTraits {
    /// Value-qualified type
    using value_type = T;
    /// unqualified type of th
    using element_type = std::remove_cvref_t<T>;

    using extents_type = Extents;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    /// coallesces  traits
    constexpr static AccessFeatures access_features = AF;
    constexpr static ShapeFeatures shape_features = SF;

    consteval static auto is_const_valued() -> bool {
        return access_features.is_const;
    }
    consteval static auto is_assignable() -> bool {
        return !is_const_valued() && is_referrable();
    }
    consteval static auto is_referrable() -> bool {
        return access_features.is_reference;
    }
    consteval static auto is_resizable() -> bool {
        return shape_features.is_resizable;
    }

    /// Coefficient-consistent means iterating over coefficients gives correct
    /// results without aliasing issues (i.e. writing a[i] doesn't affect a[j]).
    /// Default: non-reference types are alias-free, reference types may alias.
    constexpr static bool is_coefficient_consistent = !AF.is_reference;

    constexpr static bool is_writable = is_assignable();

    /// Whether this expression (or any sub-expression) stores references to
    /// external data.  When true, the expression tree may dangle if it outlives
    /// the referenced objects, so returning it from a function is disallowed
    /// by default (see NonReturnable).  Leaf expressions that own their data
    /// (e.g. MDArray, Constant) set this to false.  The default for
    /// BasicExpressionTraits is false — derived traits must propagate or
    /// override as appropriate.
    constexpr static bool stores_references = false;

    /// Whether this expression has structurally known zero regions.
    /// When true, the expression provides `index_set<D>(other_indices...)`
    /// returning an `IndexSet`-satisfying type, plus convenience aliases
    /// (`col_range_for_row`, `row_range_for_col`, `nonzero_segment`).
    /// This enables zero-aware optimizations in addition, subtraction, and
    /// matrix products.
    constexpr static bool has_index_set = false;

    /// Backward-compatible alias for has_index_set.
    constexpr static bool has_known_zeros = has_index_set;

    /// Whether copying this expression propagates view semantics.
    /// When true, `member_child_storage_t` and `expression_storage_t` store
    /// this expression by value even from lvalue context, so that derived
    /// views (head, tail, row, transpose, etc.) are also Returnable.
    ///
    /// Only UnsafeRef<Child, /*ViewPropagating=*/true> sets this to true.
    /// Unary and binary expressions propagate it from their children.
    constexpr static bool is_view_propagating = false;

    /// Layout preference for smart eval().
    /// Leaf expressions override this; non-leaf expressions propagate.
    /// Default: no preference → eval() produces default row-major dense.
    using preferred_layout = zipper::detail::NoLayoutPreference;
};

template <typename T>
struct ExpressionTraits;

/// Primary template for expression-specific implementation details.
///
/// ExpressionDetail<T> holds class-specific type computation helpers and
/// implementation details that are NOT part of the public traits interface.
/// ExpressionTraits<T> provides the standardized capability flags and
/// value_type/extents_type, while ExpressionDetail<T> holds everything
/// the expression class body needs for its internal implementation
/// (e.g., child traits aliases, extents computation utilities, index
/// mappings, rank constants).
///
/// Expression class bodies access these via:
///   using detail_type = ExpressionDetail<self_type>;
///   using lhs_traits = detail_type::ATraits;  // instead of traits::ATraits
///
/// The default (unspecialized) ExpressionDetail is an empty struct.
/// Expression types that need class-specific helpers provide specializations.
template <typename T>
struct ExpressionDetail {};

/// Const-forwarding: detail for const-qualified expressions forwards to the
/// unqualified version.
template <typename T>
struct ExpressionDetail<const T> : public ExpressionDetail<T> {};

/// Reference-forwarding: detail for reference-qualified expressions forwards
/// to the unqualified version.
template <typename T>
struct ExpressionDetail<T &> : public ExpressionDetail<T> {};
template <typename T>
struct ExpressionDetail<T &&> : public ExpressionDetail<T> {};

/// Const-forwarding: traits for const-qualified expressions forward to the
/// unqualified version, since constness doesn't change expression structure.
template <typename T>
struct ExpressionTraits<const T> : public ExpressionTraits<T> {};

/// Reference-forwarding: traits for reference-qualified expressions forward
/// to the unqualified version, since reference-ness doesn't change expression
/// structure. These are needed when expression nodes store children by
/// reference (e.g. `Diagonal<const MDArray&>`).
template <typename T>
struct ExpressionTraits<T &> : public ExpressionTraits<T> {};
template <typename T>
struct ExpressionTraits<T &&> : public ExpressionTraits<T> {};

/// Detects whether an ExpressionTraits type explicitly defines
/// is_coefficient_consistent (a non-nullary expression property).
/// For nullary expressions that don't define it, falls back to
/// !access_features.is_reference (non-reference types don't alias).
template <typename ET>
concept HasCoefficientConsistency = requires {
    { ET::is_coefficient_consistent } -> std::convertible_to<bool>;
};

template <typename ET>
consteval auto get_is_coefficient_consistent() -> bool {
    if constexpr (HasCoefficientConsistency<ET>) {
        return ET::is_coefficient_consistent;
    } else {
        return !ET::access_features.is_reference;
    }
}

/// Detects whether an expression type has structurally known zero regions.
/// Uses the ExpressionTraits<T>::has_index_set flag.
template <typename T>
concept HasIndexSet = ExpressionTraits<std::remove_cvref_t<T>>::has_index_set;

/// Backward-compatible alias for the old concept name.
template <typename T>
concept HasKnownZeros = HasIndexSet<T>;

} // namespace zipper::expression::detail
#endif
