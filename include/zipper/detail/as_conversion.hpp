#if !defined(ZIPPER_DETAIL_AS_CONVERSION_HPP)
#define ZIPPER_DETAIL_AS_CONVERSION_HPP
// Machinery of the as_* conversions (as_array, as_vector, ...).
//
// Each base header defines its own conversion with ZIPPER_DEFINE_AS after the
// base, so as_X exists exactly when XBase does.  A conversion accepts zipper
// wrappers and raw expressions (e.g. the results of over() and map()):
// lvalues are referenced (writably if the source is writable and non-const),
// rvalues are owned by the result.

#include "../concepts/Expression.hpp"
#include "../concepts/Zipper.hpp"
#include "../expression/concepts/capabilities.hpp"
#include "../expression/unary/Slice.hpp"
#include "member_child_storage.hpp"

#include <type_traits>
#include <utility> // std::in_place

namespace zipper::detail {

// ---- Squeezing --------------------------------------------------------------
//
// Converting to a base of fixed rank (as_vector, as_matrix) collapses
// dimensions whose *static* extent is 1 by slicing them at index 0, so e.g. a
// Matrix<double, 3, 1> converts to a vector.  Dynamic dimensions are never
// collapsed.

/// The number of dimensions whose static extent is not 1.
template <typename Extents>
consteval auto squeeze_rank() -> rank_type {
  rank_type count = 0;
  for (rank_type d = 0; d < Extents::rank(); ++d) {
    if (Extents::static_extent(d) != 1) ++count;
  }
  return count;
}

/// static_index_t<0> for a static extent of 1 (collapsed), else full_extent_t.
template <typename Extents, rank_type D>
using squeeze_specifier_t = std::conditional_t<Extents::static_extent(D) == 1,
                                               static_index_t<0>,
                                               full_extent_t>;

/// How a base built from T&& (a wrapper or a raw expression) holds its child.
template <typename T> struct as_child;

template <typename T>
  requires concepts::Zipper<std::remove_cvref_t<T>>
struct as_child<T> {
  using W = std::remove_reference_t<T>;
  // Const when the wrapper references a const expression.
  using Expr = std::remove_reference_t<decltype(std::declval<W &>().expression())>;
  using type = std::conditional_t<
      !std::is_lvalue_reference_v<T>,
      member_child_storage_t<W, typename W::raw_expression_type>,
      std::conditional_t<
          std::is_const_v<W>,
          member_child_storage_t<W &, typename W::expression_type>,
          member_child_storage_t<
              W &, std::conditional_t<
                       expression::concepts::WritableExpression<Expr>, Expr,
                       const Expr>>>>;
};

template <typename T>
  requires(!concepts::Zipper<std::remove_cvref_t<T>>
           && concepts::Expression<std::remove_cvref_t<T>>)
struct as_child<T> {
  using E = std::remove_reference_t<T>;
  using type = std::conditional_t<
      !std::is_lvalue_reference_v<T>, std::remove_cv_t<E>,
      std::conditional_t<!std::is_const_v<E>
                             && expression::concepts::WritableExpression<
                                 std::remove_cv_t<E>>,
                         E &, const E &>>;
};

template <typename T> using as_child_t = typename as_child<T>::type;

template <typename T> auto as_source_expression(T &&t) -> decltype(auto) {
  if constexpr (concepts::Zipper<std::remove_cvref_t<T>>) {
    return std::forward<T>(t).expression();
  } else {
    return std::forward<T>(t);
  }
}

template <typename T>
using as_extents_t = typename std::remove_cvref_t<
    decltype(as_source_expression(std::declval<T>()))>::extents_type;

/// Whether T converts to a base of rank Rank (0: any rank).
template <typename T, rank_type Rank>
concept ConvertibleToRank =
    (concepts::Zipper<std::remove_cvref_t<T>>
     || concepts::Expression<std::remove_cvref_t<T>>)
    && (Rank == 0 || as_extents_t<T>::rank() == Rank
        || squeeze_rank<as_extents_t<T>>() == Rank);

/// `t` as Base; for a base of rank Rank, extra statically degenerate
/// dimensions are squeezed away.
template <template <typename> typename Base, rank_type Rank, typename T>
  requires ConvertibleToRank<T, Rank>
auto as_base(T &&t) {
  using Child = as_child_t<T &&>;
  using extents_type = as_extents_t<T>;
  if constexpr (Rank == 0 || extents_type::rank() == Rank) {
    return Base<Child>(std::in_place, as_source_expression(std::forward<T>(t)));
  } else {
    return [&]<rank_type... Ds>(std::integer_sequence<rank_type, Ds...>) {
      using SliceT = expression::unary::Slice<
          Child, squeeze_specifier_t<extents_type, Ds>...>;
      return Base<SliceT>(std::in_place,
                          as_source_expression(std::forward<T>(t)),
                          squeeze_specifier_t<extents_type, Ds>{}...);
    }(std::make_integer_sequence<rank_type, extents_type::rank()>{});
  }
}

} // namespace zipper::detail

/// Defines zipper::as_NAME, converting wrappers and raw expressions to BASE;
/// RANK is the base's rank, or 0 for any.  Use in namespace zipper, after
/// BASE.
#define ZIPPER_DEFINE_AS(NAME, BASE, RANK)                                     \
  template <typename T>                                                        \
    requires detail::ConvertibleToRank<T, RANK>                                \
  auto as_##NAME(T &&t) {                                                      \
    return detail::as_base<BASE, RANK>(std::forward<T>(t));                    \
  }

#endif
