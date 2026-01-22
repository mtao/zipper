
#if !defined(ZIPPER_EXPRESSION_NULLARY_CONSTANT_HPP)
#define ZIPPER_EXPRESSION_NULLARY_CONSTANT_HPP

#include "NullaryExpressionBase.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"

namespace zipper::expression {
namespace nullary {
template <typename T, index_type... Indices> class Constant;

}
template <typename T, index_type... Indices>
struct detail::ExpressionTraits<nullary::Constant<T, Indices...>>
    : public BasicExpressionTraits<
          T, zipper::extents<Indices...>,
          expression::detail::AccessFeatures{
              .is_const = false, .is_reference = false, .is_alias_free = true},
          expression::detail::ShapeFeatures{.is_resizable = true}> {};

}

constexpr static bool is_plain_data = false;
using base_type =
    SizedExpressionBase<typename nullary::Constant<T, Indices...>>;
}
;

namespace nullary {
template <typename T, index_type... Indices>
class Constant
    : public NullaryExpressionBase<Constant<T, Indices...>, T, Indices...> {
public:
  using self_type = Constant<T, Indices...>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = traits::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  using value_type = traits::value_type;
  using Base = NullaryExpressionBase<Constant<T, Indices...>, T, Indices...>;

  Constant(const value_type &v, const extents_type &e = {})
      : Base(e), m_value(v) {}

  auto get_value() const -> value_type { return m_value; }

private:
  value_type m_value;
};

template <typename T, index_type... Indices>
Constant(const T &, const extents<Indices...> &) -> Constant<T, Indices...>;
template <typename T> Constant(const T &) -> Constant<T>;

} // namespace nullary
} // namespace zipper::expression
#endif
