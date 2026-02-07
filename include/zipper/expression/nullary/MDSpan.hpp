#if !defined(ZIPPER_EXPRESSION_NULLARY_MDSPAN_HPP)
#define ZIPPER_EXPRESSION_NULLARY_MDSPAN_HPP

#include "LinearLayoutExpression.hpp"
#include "zipper/detail//ExtentsTraits.hpp"
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
          Extents, LayoutPolicy, AccessorPolicy> {

  using base_type = LinearLayoutExpression<
      storage::SpanData<ElementType,
                        zipper::detail::ExtentsTraits<Extents>::static_size>,
      Extents, LayoutPolicy, AccessorPolicy>;
  using base_type::base_type;
};
} // namespace zipper::expression::nullary

#endif
