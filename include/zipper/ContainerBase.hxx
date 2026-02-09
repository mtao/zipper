#if !defined(ZIPPER_CONTAINERBASE_HXX)
#define ZIPPER_CONTAINERBASE_HXX

#include "ContainerBase.hpp"
#include "expression/nullary/MDSpan.hpp"
#include "utils/extents/all_extents_indices.hpp"
#include <compare>
#include <tuple>

namespace zipper {

// Deduction guide from std::mdspan
template <typename T, typename Extents, typename Layout, typename Accessor>
ContainerBase(zipper::mdspan<T, Extents, Layout, Accessor>) -> ContainerBase<
    expression::nullary::MDSpan<T, Extents, Layout, Accessor>>;

// Deduction guides from std::span
template <typename T, std::size_t N>
ContainerBase(std::span<T, N>) -> ContainerBase<
    expression::nullary::MDSpan<T, extents<N>>>;

template <typename T>
ContainerBase(std::span<T, std::dynamic_extent>) -> ContainerBase<
    expression::nullary::MDSpan<T, extents<dynamic_extent>>>;

// Deduction guides from std::vector
template <typename T, typename Alloc>
ContainerBase(std::vector<T, Alloc> &) -> ContainerBase<
    expression::nullary::MDSpan<T, extents<dynamic_extent>>>;

template <typename T, typename Alloc>
ContainerBase(const std::vector<T, Alloc> &) -> ContainerBase<
    expression::nullary::MDSpan<const T, extents<dynamic_extent>>>;

// Deduction guides from std::array
template <typename T, std::size_t N>
ContainerBase(std::array<T, N> &) -> ContainerBase<
    expression::nullary::MDSpan<T, extents<N>>>;

template <typename T, std::size_t N>
ContainerBase(const std::array<T, N> &) -> ContainerBase<
    expression::nullary::MDSpan<const T, extents<N>>>;

// Lexicographic three-way comparison (std::weak_ordering).
// Two containers compare equal iff they have the same extents
// and all elements compare equal in index order.  Otherwise the
// first differing element determines the ordering.
// Containers with different ranks are ordered by rank first.
// Containers with same rank but different extents are ordered
// by extents lexicographically first, then by elements.
template <concepts::Container Expr1, concepts::Container Expr2>
auto operator<=>(const Expr1 &lhs, const Expr2 &rhs) -> std::weak_ordering
  requires(Expr1::extents_type::rank() == Expr2::extents_type::rank())
{
  constexpr rank_type rank = Expr1::extents_type::rank();
  const auto lhs_ext = lhs.extents();
  const auto rhs_ext = rhs.extents();

  // First compare extents lexicographically
  for (rank_type r = 0; r < rank; ++r) {
    if (lhs_ext.extent(r) < rhs_ext.extent(r))
      return std::weak_ordering::less;
    if (lhs_ext.extent(r) > rhs_ext.extent(r))
      return std::weak_ordering::greater;
  }

  // Same extents — compare elements in index order
  for (auto &&idx : zipper::utils::extents::all_extents_indices(lhs_ext)) {
    auto lval = std::apply(
        [&](auto... idxs) -> decltype(auto) { return lhs(idxs...); }, idx);
    auto rval = std::apply(
        [&](auto... idxs) -> decltype(auto) { return rhs(idxs...); }, idx);
    if (lval < rval)
      return std::weak_ordering::less;
    if (rval < lval)
      return std::weak_ordering::greater;
  }

  return std::weak_ordering::equivalent;
}

template <concepts::Container Expr1, concepts::Container Expr2>
auto operator==(const Expr1 &lhs, const Expr2 &rhs) -> bool
  requires(Expr1::extents_type::rank() == Expr2::extents_type::rank())
{
  return (lhs <=> rhs) == std::weak_ordering::equivalent;
}

} // namespace zipper

#endif
