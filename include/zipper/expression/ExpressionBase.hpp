#if !defined(ZIPPER_EXPRESSION_EXPRESSIONBASE_HPP)
#define ZIPPER_EXPRESSION_EXPRESSIONBASE_HPP

#include "detail/ExpressionTraits.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/concepts/IndexArgument.hpp"
#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/types.hpp"

namespace zipper::expression {

/** The CRTP base class for every zipper view
 *
 * ## Extents
 * - extents() -> extents_type
 * - extent(rank_type) -> index_type
 *   size() const -> index_type
 *   coeff(Index...) const -> value_type;
 *
 * - operator(Args...) const -> const value_type&
 * - operator(Args...) const -> value_type&
 */
template <typename Derived_> class ExpressionBase {
public:
  using Derived = Derived_;

  auto derived() -> Derived & { return static_cast<Derived &>(*this); }
  auto derived() const -> const Derived & {
    return static_cast<const Derived &>(*this);
  }

  using traits = detail::ExpressionTraits<Derived>;

  using extents_type = traits::extents_type;
  using value_type = traits::value_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  // expression does not permute underlying value of indices, so
  // coefficient-wise operations are valid
  constexpr static bool is_alias_free = traits::is_alias_free;
  constexpr static bool is_plain_data = traits::is_plain_data;
  constexpr static bool is_const = traits::is_const;
  constexpr static rank_type rank = extents_type::rank();

  static_assert(extents_type::rank() >= 0);
  using array_type = std::array<index_type, rank>;

  [[nodiscard]] constexpr auto extent(rank_type i) const -> index_type {
    return derived().extent(i);
  }
  static consteval auto static_extent(rank_type i) -> index_type {
    return extents_type::static_extent(i);
  }
  constexpr auto extents() const -> const extents_type & {
    return derived().extents();
  }

  [[nodiscard]] constexpr auto size() const -> size_t {
    return extents_traits::size(extents());
  }

public:
  template <concepts::Index... Indices>
  auto coeff(Indices &&...indices) const -> value_type;
  template <concepts::Index... Indices>
  auto coeff_ref(Indices &&...indices) -> value_type &
    requires(is_plain_data);

  template <concepts::Index... Indices>
  auto const_coeff_ref(Indices &&...indices) const -> const value_type &
    requires(is_plain_data);

  /// Internally forwards to const_access_pack
  template <concepts::IndexArgument... Args>
  auto operator()(Args &&...idxs) const -> decltype(auto);

  /// Primary access point, because it's mutable must support is_plain_data and
  /// the underlying view is considered mutable
  /// Internally forwards to access_pack
  template <concepts::IndexArgument... Args>
  auto operator()(Args &&...idxs) -> decltype(auto)
    requires(is_plain_data && !is_const);

private:
  /// generic entrypoint for accessing values
  template <concepts::IndexArgument... Args>
  auto access_pack(Args &&...idxs) -> decltype(auto);
  template <concepts::IndexArgument... Args>
  auto const_access_pack(Args &&...idxs) const -> decltype(auto);

  template <concepts::Index... Args>
  auto access_index_pack(Args &&...idxs) -> decltype(auto);
  template <concepts::Index... Args>
  auto const_access_index_pack(Args &&...idxs) const -> decltype(auto);

  template <concepts::IndexSlice... Slices>
  auto access_slice(Slices &&...slices);
  template <concepts::IndexSlice... Slices>
  auto const_access_slice(Slices &&...slices) const;

  template <concepts::IndexArgumentPackTuple Tuple>
  auto access_tuple(const Tuple &t) -> decltype(auto);
  template <concepts::IndexArgumentPackTuple Tuple>
  auto const_access_tuple(const Tuple &t) const -> decltype(auto);
};

} // namespace zipper::expression
#if !defined(ZIPPER_expression_EXPRESSIONBASE_HXX)
#include "ExpressionBase.hxx"
#endif
#endif
