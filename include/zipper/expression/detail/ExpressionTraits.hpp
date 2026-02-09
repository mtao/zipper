#if !defined(ZIPPER_EXPRESSION_DETAIL_EXPRESSIONTRAITS_HPP)
#define ZIPPER_EXPRESSION_DETAIL_EXPRESSIONTRAITS_HPP
#include "zipper/concepts/Extents.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/detail/Features.hpp"
#include "zipper/types.hpp"

namespace zipper::expression::detail {

using AccessFeatures = zipper::detail::AccessFeatures;
using ShapeFeatures = zipper::detail::ShapeFeatures;
template <typename T, concepts::Extents Extents,
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
  consteval static auto is_reference_valued() -> bool {
    return access_features.is_reference;
  }
  consteval static auto is_assignable() -> bool {
    return !is_const_valued() && is_reference_valued();
  }
  consteval static auto is_referrable() -> bool {
    return access_features.is_reference;
  }
  consteval static auto is_resizable() -> bool {
    return shape_features.is_resizable;
  }
  consteval auto is_alias_free() -> bool {
    return access_features.is_alias_free;
  }

  /// Coefficient-consistent means iterating over coefficients gives correct
  /// results without aliasing issues (i.e. writing a[i] doesn't affect a[j]).
  constexpr static bool is_coefficient_consistent = AF.is_alias_free;

  constexpr static bool is_writable = is_assignable();
};

template <typename T> struct ExpressionTraits;

/// Const-forwarding: traits for const-qualified expressions forward to the
/// unqualified version, since constness doesn't change expression structure.
template <typename T>
struct ExpressionTraits<const T> : public ExpressionTraits<T> {};

/// This concept is designed for debug / testing that an extents traits is
/// reasonable
template <typename ET>
concept ExpressionTraitsConcept =
    std::is_same_v<std::remove_cvref_t<typename ET::value_type>,
                   typename ET::element_type> &&
    requires(ET et) {
      /// Identifies if the expression is constant
      { et.is_const_valued() } -> std::same_as<bool>;
      /// Identifies if values can be assigned or not. shape checking required
      { et.is_assignable() } -> std::same_as<bool>;
      /// Is the underlying value a reference type
      { et.is_reference_valued() } -> std::same_as<bool>;
      // can the value type be referred to
      { et.is_resizable() } -> std::same_as<bool>;
      /// Can we guarantee that if a value is assigned it only modifies a single
      /// entry
      { et.is_alias_free() } -> std::same_as<bool>;
      /// A const expression that is assignable is NOT allowed
    } && (!(ET::is_const_valued() && ET::is_assignable()));

// NOTE: template parameters should NOT be used in this struct so that derived
// can overwrite them.
template <typename ValueType = void, typename Extents = zipper::dextents<0>>
struct DefaultExpressionTraits {
  /// Value-qualified type
  using value_type = ValueType;
  /// unqualified type of th
  using element_type = std::remove_cvref_t<ValueType>;

  using extents_type = Extents;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

  // New-style feature structs required by ExpressionBase
  constexpr static AccessFeatures access_features = {
      .is_const = std::is_const_v<value_type>,
      .is_reference = false,
      /// guarantees that V(j) = f(...) cannot depend on V(k) for j != k)
      .is_alias_free = false,
  };
  constexpr static ShapeFeatures shape_features = {
      .is_resizable = false,
  };

  consteval static auto is_const_valued() -> bool {
    return access_features.is_const;
  }
  consteval static auto is_reference_valued() -> bool {
    return access_features.is_reference;
  }
  consteval static auto is_assignable() -> bool {
    return access_features.is_assignable();
  }
  consteval static auto is_referrable() -> bool {
    return is_reference_valued();
  }

  consteval static auto is_resizable() -> bool {
    return shape_features.is_resizable;
  }

  /// Coefficient-consistent means iterating over coefficients gives correct
  /// results without aliasing issues.
  constexpr static bool is_coefficient_consistent =
      access_features.is_alias_free;

  constexpr static bool is_writable = is_assignable();

  /// returns true if a dimension is sparse, false if dense. Should be used to
  /// hint if a dimension should be iterated via a full loop or an iterator
  /// (TODO: determine an iterator interface)
  consteval static auto is_sparse(rank_type) -> bool { return false; }
};

} // namespace zipper::expression::detail
#endif
