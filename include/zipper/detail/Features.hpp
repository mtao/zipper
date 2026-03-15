#if !defined(ZIPPER_DETAIL_FEATURES_HPP)
#define ZIPPER_DETAIL_FEATURES_HPP
#include <type_traits>

namespace zipper::detail {

struct AccessFeatures {
  /// Is the underlying value mutable / const
  bool is_const = false;
  /// Is the underlying type a reference type
  bool is_reference = false;
  template <typename ValueType>
  constexpr static auto from_type() -> AccessFeatures {
    return {
        .is_const = std::is_const_v<ValueType>,
        .is_reference = std::is_reference_v<ValueType>,
    };
  }

  constexpr auto operator||(AccessFeatures o) -> AccessFeatures {
    return AccessFeatures{
        .is_const = is_const || o.is_const,
        .is_reference = is_reference || o.is_reference,
    };
  }

  [[nodiscard]] constexpr auto is_assignable() const -> bool {
    return !is_const && is_reference;
  }
};

struct ShapeFeatures {
  /// whether we can resize the expression if it has dynamic indices
  bool is_resizable = false;
};

} // namespace zipper::detail

#endif
