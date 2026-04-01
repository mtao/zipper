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
    return {std::is_const_v<ValueType>, std::is_reference_v<ValueType>};
  }

  constexpr auto operator||(AccessFeatures o) -> AccessFeatures {
    return AccessFeatures{is_const || o.is_const,
                          is_reference || o.is_reference};
  }

  [[nodiscard]] constexpr auto is_assignable() const -> bool {
    return !is_const && is_reference;
  }

  // Named factories for common NTTP patterns (avoids designated initializers
  // which MSVC rejects in template argument positions)
  constexpr static auto const_value() -> AccessFeatures {
    return {true, false};
  }
  constexpr static auto mutable_value() -> AccessFeatures {
    return {false, false};
  }
  constexpr static auto const_ref() -> AccessFeatures {
    return {true, true};
  }
  constexpr static auto mutable_ref() -> AccessFeatures {
    return {false, true};
  }
};

struct ShapeFeatures {
  /// whether we can resize the expression if it has dynamic indices
  bool is_resizable = false;

  // Named factories for common NTTP patterns
  constexpr static auto fixed() -> ShapeFeatures { return {false}; }
  constexpr static auto resizable() -> ShapeFeatures { return {true}; }
};

} // namespace zipper::detail

#endif
