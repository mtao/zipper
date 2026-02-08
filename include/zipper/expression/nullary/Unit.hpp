#if !defined(ZIPPER_EXPRESSION_NULLARY_UNIT_HPP)
#define ZIPPER_EXPRESSION_NULLARY_UNIT_HPP

#include <utility>

#include "zipper/expression/ExpressionBase.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"

namespace zipper::expression {
namespace nullary {

/// A unit vector expression: returns 1 at a single index and 0 elsewhere.
///
/// @tparam T         Value type
/// @tparam Extent    Static extent (or dynamic_extent)
/// @tparam IndexType Either index_type (dynamic index) or
///                   std::integral_constant<index_type, I> (static index)
template <typename T, index_type Extent, typename IndexType>
class Unit : public ExpressionBase<Unit<T, Extent, IndexType>>,
             public zipper::extents<Extent> {
public:
  using self_type = Unit<T, Extent, IndexType>;
  using traits = expression::detail::ExpressionTraits<self_type>;
  using extents_type = typename traits::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  using value_type = typename traits::value_type;

  using extents_type::extent;
  using extents_type::rank;
  auto extents() const -> const extents_type & { return *this; }

  constexpr static bool dynamic_index =
      std::is_same_v<IndexType, index_type>;
  constexpr static bool dynamic_size = extents_traits::is_dynamic;

  // If the size is dynamic we require a dynamic index
  static_assert(dynamic_index || !dynamic_size);

  Unit(const Unit &) = default;
  Unit(Unit &&) = default;
  auto operator=(const Unit &) -> Unit & = default;
  auto operator=(Unit &&) -> Unit & = default;

  /// Static extent + static index
  Unit()
    requires(extents_traits::is_static && !dynamic_index)
      : m_index() {}

  /// Static extent + dynamic index
  Unit(const index_type index)
    requires(!dynamic_size && dynamic_index)
      : m_index(index) {}

  /// Dynamic extent + index
  Unit(const index_type my_extent, IndexType index)
      : extents_type(my_extent), m_index(index) {}

  template <concepts::Index Idx>
  auto coeff(const Idx &idx) const -> value_type {
    if (static_cast<index_type>(idx) == static_cast<index_type>(m_index)) {
      return value_type(1);
    } else {
      return value_type(0);
    }
  }

  template <rank_type R>
    requires(R == 0)
  constexpr auto nonZeros(index_type) const -> std::array<index_type, 1> {
    return std::array<index_type, 1>{{static_cast<index_type>(m_index)}};
  }

private:
  IndexType m_index;
};

/// Static extent + static index
template <typename T, index_type size, index_type index>
auto unit_vector() {
  return Unit<T, size, std::integral_constant<index_type, index>>{};
}

/// Dynamic extent + dynamic index
template <typename T>
auto unit_vector(index_type size, index_type index) {
  return Unit<T, std::dynamic_extent, index_type>{size, index};
}

/// Static extent + dynamic index
template <typename T, index_type size>
auto unit_vector(index_type index)
  requires(size != std::dynamic_extent)
{
  return Unit<T, size, index_type>(index);
}

} // namespace nullary

template <typename T, index_type Extent, typename IndexType>
struct detail::ExpressionTraits<nullary::Unit<T, Extent, IndexType>>
    : public BasicExpressionTraits<
          T, zipper::extents<Extent>,
          expression::detail::AccessFeatures{
              .is_const = false, .is_reference = false, .is_alias_free = true},
          expression::detail::ShapeFeatures{.is_resizable = false}> {

  constexpr static bool is_value_based = false;

  // Unit vectors are sparse
  consteval static auto is_sparse(rank_type) -> bool { return true; }
};

} // namespace zipper::expression
#endif
