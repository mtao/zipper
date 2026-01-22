#if !defined(ZIPPER_EXPRESSION_DETAIL_EXPRESSIONTRAITS_HPP)
#define ZIPPER_EXPRESSION_DETAIL_EXPRESSIONTRAITS_HPP
#include "zipper/concepts/Extents.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/types.hpp"

namespace zipper::expression::detail {
struct AccessFeatures {
  /// Is the underlying value mutable / const
  bool is_const = false;
  /// Is the underlying type a reference type
  bool is_reference = false;
  /// can modifying one element modify another element in the span
  bool is_alias_free = false;
  template <typename ValueType>
  constexpr static auto from_type() -> AccessFeatures {
    return {
        .is_const = std::is_const_v<ValueType>,
        .is_reference = std::is_reference_v<ValueType>,
        // if the input
        // type is a ref then we can be pretty sure it can't prevent aliasing
        .is_alias_free = !std::is_reference_v<ValueType>};
  }

  constexpr auto operator||(AccessFeatures o) -> AccessFeatures {
    return AccessFeatures{
        .is_const = is_const || o.is_const,
        .is_reference = is_reference || o.is_reference,
        .is_alias_free = is_alias_free || o.is_alias_free,
    };
  }
};

struct ShapeFeatures {
  /// whether we can resize the expression if it has dynamic indices
  bool is_resizable = false;
};
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

  consteval auto is_const_valued() -> bool { return access_features.is_const; }
  consteval auto is_reference_valued() -> bool {
    return access_features.is_reference;
  }
  consteval auto is_assignable() -> bool {
    return !is_const_valued() && is_reference_valued();
  }
  consteval auto is_referrable() -> bool {
    return access_features.is_reference;
  }
  consteval auto is_resizable() -> bool { return shape_features.is_resizable; }
  consteval auto is_alias_free() -> bool {
    return access_features.is_alias_free;
  }
};

template <typename T> struct ExpressionTraits;

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
    } && (!(ET::is_const() && ET::is_assignable()));

template <typename T> struct ExpressionTraits;

// NOTE: template parameters should NOT be used in this struct so that derived
// can overwrite them
template <typename ValueType = void, typename Extents = zipper::dextents<0>>
struct DefaultExpressionTraits {
  /// Value-qualified type
  using value_type = ValueType;
  /// unqualified type of th
  using element_type = std::remove_cvref_t<ValueType>;

  using extents_type = Extents;

  constexpr static bool is_const = std::is_const_v<value_type>;
  constexpr static bool is_assignable = false;

  constexpr static bool is_resizable = false;

  /// guarantees that V(j) = f(...) cannot depend on V(k) for j != k)
  constexpr static bool is_alias_free = false;

  /// returns true if a dimension is sparse, false if dense. Should be used to
  /// hint if a dimension should be iterated via a full loop or an iterator
  /// (TODO: determine an iterator interface)
  consteval static auto is_sparse(rank_type) -> bool { return false; }
};

} // namespace zipper::expression::detail
#endif
