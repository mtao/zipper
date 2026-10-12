#if !defined(ZIPPER_EXPRESSION_NULLARY_MDARRAY_HPP)
#define ZIPPER_EXPRESSION_NULLARY_MDARRAY_HPP

#include "zipper/detail/NoAliasTraits.hpp"
#include "LinearLayoutExpression.hpp"
#include "MDSpan.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"
#include "zipper/storage/DenseData.hpp"
#include "zipper/storage/layout_types.hpp"

namespace zipper::expression::nullary {
template <typename ElementType, typename Extents,
          typename LayoutPolicy = default_layout_policy>
class MDArray
    : public LinearLayoutExpression<
          storage::DenseData<
              ElementType, zipper::detail::ExtentsTraits<Extents>::static_size>,
          Extents, LayoutPolicy,
          MDArray<ElementType, Extents, LayoutPolicy>> {

  using base_type = LinearLayoutExpression<
      storage::DenseData<ElementType,
                         zipper::detail::ExtentsTraits<Extents>::static_size>,
      Extents, LayoutPolicy,
      MDArray<ElementType, Extents, LayoutPolicy>>;
  using base_type::base_type;

  constexpr static bool IsStatic = base_type::extents_traits::is_static;
  using span_type = MDSpan<ElementType, Extents, LayoutPolicy>;
public:
  using typename base_type::extents_traits;
  using typename base_type::extents_type;
  using base_type::linear_accessor;
  using self_type = MDArray<ElementType, Extents, LayoutPolicy>;

  template <zipper::concepts::Expression V>
  void assign(const V &v)
    requires(zipper::utils::extents::assignable_extents_v<
                typename V::extents_type, extents_type>)
  {
    expression::detail::AssignHelper<V, self_type>::assign(v, *this);
  }

  /// MDArray already owns its data — make_owned() returns a copy.
  auto make_owned() const -> self_type { return *this; }

  auto as_span() -> span_type {
    if constexpr (IsStatic) {
      return span_type(linear_accessor().as_std_span());
    } else {
      const extents_type &e = this->extents();
      return span_type(linear_accessor().as_std_span(), e);
    }
  }
  auto as_span() const -> const span_type {
    if constexpr (IsStatic) {
      return span_type(linear_accessor().as_std_span());
    } else {
      const extents_type &e = this->extents();
      return span_type(linear_accessor().as_std_span(), e);
    }
  }

  template <zipper::concepts::Extents E2>
  void resize(const E2 &e)
    requires(extents_traits::template is_convertable_from<E2>() && !IsStatic &&
             E2::rank() != 0)
  {
    this->resize_extents(e);
    linear_accessor().container().resize(
        zipper::detail::ExtentsTraits<E2>::size(e));
  }

  /// Resize without initializing newly added elements. See
  /// zipper::uninitialized_t.
  template <zipper::concepts::Extents E2>
  void resize(uninitialized_t, const E2 &e)
    requires(extents_traits::template is_convertable_from<E2>() && !IsStatic &&
             E2::rank() != 0 && base_type::uninitialized_constructible)
  {
    this->resize_extents(e);
    linear_accessor().container().resize(
        uninitialized, zipper::detail::ExtentsTraits<E2>::size(e));
  }
};

} // namespace zipper::expression::nullary

namespace zipper::expression {
/// ExpressionTraits for MDArray forwards to the LinearLayoutExpression traits,
/// since MDArray is a thin owning wrapper around LinearLayoutExpression.
template <typename ElementType, typename Extents, typename LayoutPolicy>
struct detail::ExpressionTraits<
    nullary::MDArray<ElementType, Extents, LayoutPolicy>>
    : public detail::ExpressionTraits<nullary::LinearLayoutExpression<
          storage::DenseData<ElementType,
                             zipper::detail::ExtentsTraits<Extents>::static_size>,
          Extents, LayoutPolicy,
          nullary::MDArray<ElementType, Extents, LayoutPolicy>>> {
};
} // namespace zipper::expression

// Out-of-line definition of MDSpan::make_owned() — must appear after MDArray
// is fully defined because it constructs an MDArray.
namespace zipper::expression::nullary {
template <typename ElementType, typename Extents,
          typename LayoutPolicy, typename AccessorPolicy>
auto MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>::make_owned() const {
  using non_const_element = std::remove_const_t<ElementType>;
  // Strided views copy into the default dense layout.
  using owned_type = MDArray<non_const_element, Extents,
                             owning_layout_policy_t<LayoutPolicy, Extents>>;
  // evaluate_to overwrites every coefficient: skip the zero-fill.
  owned_type result(zipper::uninitialized, this->extents());
  expression::detail::AssignHelper<self_type, owned_type>::evaluate_to(*this, result);
  return result;
}
} // namespace zipper::expression::nullary

namespace zipper::detail {
// Mutable dense owning storage supports noalias destinations.
template <typename T, typename E, typename L>
inline constexpr bool
    supports_noalias_v<expression::nullary::MDArray<T, E, L>> =
        !std::is_const_v<T>;
} // namespace zipper::detail

#endif
