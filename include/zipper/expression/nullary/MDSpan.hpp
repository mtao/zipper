#if !defined(ZIPPER_EXPRESSION_NULLARY_MDSPAN_HPP)
#define ZIPPER_EXPRESSION_NULLARY_MDSPAN_HPP

#include "LinearLayoutExpression.hpp"
#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"
#include "zipper/storage/SpanData.hpp"
#include "zipper/storage/layout_types.hpp"

namespace zipper::expression::nullary {

template <typename ElementType, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<ElementType>>
class MDSpan
    : public LinearLayoutExpression<
          storage::SpanData<
              ElementType, zipper::detail::ExtentsTraits<Extents>::static_size>,
          Extents, LayoutPolicy, AccessorPolicy,
          MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>> {

  using base_type = LinearLayoutExpression<
      storage::SpanData<ElementType,
                        zipper::detail::ExtentsTraits<Extents>::static_size>,
      Extents, LayoutPolicy, AccessorPolicy,
      MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>>;
  using base_type::base_type;

public:
  using self_type = MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
  using extents_type = Extents;

  template <concepts::Expression V>
  void assign(const V &v)
    requires(!std::is_const_v<ElementType> &&
             zipper::utils::extents::assignable_extents_v<
                 typename V::extents_type, extents_type>)
  {
    expression::detail::AssignHelper<V, self_type>::assign(v, *this);
  }
};
} // namespace zipper::expression::nullary

namespace zipper::expression {
/// ExpressionTraits for MDSpan forwards to the LinearLayoutExpression traits.
template <typename ElementType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ExpressionTraits<nullary::MDSpan<
    ElementType, Extents, LayoutPolicy, AccessorPolicy>>
    : public detail::ExpressionTraits<nullary::LinearLayoutExpression<
          storage::SpanData<ElementType,
                            zipper::detail::ExtentsTraits<Extents>::static_size>,
          Extents, LayoutPolicy, AccessorPolicy,
          nullary::MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>>> {
};
} // namespace zipper::expression

#endif
