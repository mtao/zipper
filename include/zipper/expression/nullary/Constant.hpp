
#if !defined(ZIPPER_EXPRESSION_NULLARY_CONSTANT_HPP)
#define ZIPPER_EXPRESSION_NULLARY_CONSTANT_HPP

#include "NullaryExpressionBase.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"

namespace zipper::expression {
namespace nullary {
template <typename T, index_type... Indices>
class Constant : public NullaryExpressionBase<Constant<T, Indices...>>,
                 public zipper::extents<Indices...> {
public:
  using self_type = Constant<T, Indices...>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = typename traits::extents_type;
  using extents_traits = typename traits::extents_traits;
  using value_type = typename traits::value_type;
  using nullary_base_type = NullaryExpressionBase<self_type>;

  using extents_type::extent;
  using extents_type::rank;
  auto extents() const -> const extents_type & { return *this; }

  Constant(const value_type &v, const extents_type &e = {})
      : extents_type(e), m_value(v) {}

  /// Constructs a constant using value_type{}
  Constant(const extents_type &e) : extents_type(e) {}

  Constant()
    requires(extents_traits::is_const)
  = default;
  Constant(const Constant &) = default;
  Constant(Constant &&) = default;
  auto operator=(const Constant &) -> Constant & = default;
  auto operator=(Constant &&) -> Constant & = default;

  auto get_value() const -> value_type { return m_value; }

private:
  value_type m_value = {};
};

template <typename T, index_type... Indices>
Constant(const T &, const extents<Indices...> &) -> Constant<T, Indices...>;
template <typename T> Constant(const T &) -> Constant<T>;

} // namespace nullary
template <typename T, index_type... Indices>
struct detail::ExpressionTraits<nullary::Constant<T, Indices...>>
    : public BasicExpressionTraits<
          T, zipper::extents<Indices...>,
          expression::detail::AccessFeatures{
              .is_const = false, .is_reference = false, .is_alias_free = true},
          expression::detail::ShapeFeatures{.is_resizable = true}> {};

} // namespace zipper::expression
#endif
