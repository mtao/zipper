#if !defined(ZIPPER_EXPRESSION_NULLARY_MDARRAY_HPP)
#define ZIPPER_EXPRESSION_NULLARY_MDARRAY_HPP

#include "LinearLayoutExpression.hpp"
#include "MDSpan.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"
#include "zipper/storage/DenseData.hpp"
#include "zipper/storage/layout_types.hpp"

namespace zipper::expression::nullary {
template <typename ElementType, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<ElementType>>
class MDArray
    : public LinearLayoutExpression<
          storage::DenseData<
              ElementType, zipper::detail::ExtentsTraits<Extents>::static_size>,
          Extents, LayoutPolicy, AccessorPolicy,
          MDArray<ElementType, Extents, LayoutPolicy, AccessorPolicy>> {

  using base_type = LinearLayoutExpression<
      storage::DenseData<ElementType,
                         zipper::detail::ExtentsTraits<Extents>::static_size>,
      Extents, LayoutPolicy, AccessorPolicy,
      MDArray<ElementType, Extents, LayoutPolicy, AccessorPolicy>>;
  using base_type::base_type;

  constexpr static bool IsStatic = base_type::extents_traits::is_static;
  using span_type = MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
public:
  using typename base_type::extents_traits;
  using typename base_type::extents_type;
  using base_type::linear_accessor;
  using self_type = MDArray<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

  template <concepts::Expression V>
  void assign(const V &v)
    requires(zipper::utils::extents::assignable_extents_v<
                typename V::extents_type, extents_type>)
  {
    expression::detail::AssignHelper<V, self_type>::assign(v, *this);
  }

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
    requires(extents_traits::template is_convertable_from<E2>() && !IsStatic)
  {
    static_assert(E2::rank() != 0);
    this->resize_extents(e);
    linear_accessor().container().resize(
        zipper::detail::ExtentsTraits<E2>::size(e));
  }
};

} // namespace zipper::expression::nullary

namespace zipper::expression {
/// ExpressionTraits for MDArray forwards to the LinearLayoutExpression traits,
/// since MDArray is a thin owning wrapper around LinearLayoutExpression.
template <typename ElementType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ExpressionTraits<nullary::MDArray<
    ElementType, Extents, LayoutPolicy, AccessorPolicy>>
    : public detail::ExpressionTraits<nullary::LinearLayoutExpression<
          storage::DenseData<ElementType,
                             zipper::detail::ExtentsTraits<Extents>::static_size>,
          Extents, LayoutPolicy, AccessorPolicy,
          nullary::MDArray<ElementType, Extents, LayoutPolicy, AccessorPolicy>>> {
};
} // namespace zipper::expression
#endif
