#if !defined(ZIPPER_EXPRESSION_NULLARY_DENSESTORAGEEXPRESSIONBASE_HPP)
#define ZIPPER_EXPRESSION_NULLARY_DENSESTORAGEEXPRESSIONBASE_HPP

#include "detail/DenseStorageExpressionTraits.hpp"
#include "zipper/expression/MappedExpressionBase.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
namespace zipper::expression::nullary {
template <typename Derived_>
class DenseStorageExpressionBase : public MappedExpressionBase<Derived_> {
public:
  using Derived = Derived_;
  auto derived() -> Derived & { return static_cast<Derived &>(*this); }
  auto derived() const -> const Derived & {
    return static_cast<const Derived &>(*this);
  }
  using BaseType = ExpressionBase<Derived>;
  using traits = expression::detail::ExpressionTraits<Derived>;
  constexpr static bool is_const = traits::is_const;

  using extents_type = traits::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  using value_type = traits::value_type;
  constexpr static bool IsStatic = extents_traits::is_static;

  static_assert(extents_type::rank() > 0);
  using ParentType = MappedExpressionBase<Derived>;
  using ParentType::extent;
  using ParentType::extents;
  static_assert(
      std::is_same_v<typename ParentType::extents_type, extents_type>);

  using ParentType::ParentType;

  using layout_policy = traits::layout_policy;
  using accessor_policy = traits::accessor_policy;
  using value_accessor_type = traits::value_accessor_type;
  using mapping_type = typename layout_policy::template mapping<extents_type>;
  using span_type = extents_traits::template span_type<value_type>;
  using const_span_type = extents_traits::template span_type<const value_type>;
  using mdspan_type =
      zipper::mdspan<value_type, extents_type, layout_policy, accessor_policy>;

public:
  auto accessor() -> value_accessor_type & { return derived().linear_access(); }
  auto accessor() const -> const value_accessor_type &;
  auto data() -> value_type *
    requires(!is_const);
  auto data() const -> const value_type * { return accessor().data(); }
  auto coeff_linear(index_type i) const -> value_type;
  auto coeff_ref_linear(index_type i) -> value_type &
    requires(!is_const);
  auto const_coeff_ref_linear(index_type i) const -> const value_type &;

  auto as_mdspan() -> mdspan_type
    requires(!is_const);
  auto as_mdspan() const -> const mdspan_type;

  // todo fixing names
  auto as_std_span() -> span_type { return accessor().as_std_span(); }
  auto as_std_span() const -> const_span_type;

  template <typename E2>
  void resize(const E2 &e)
    requires(!is_const);

public:
  template <zipper::concepts::Expression V>
  void assign(const V &v)
    requires(extents_traits::template is_convertable_from<
             typename expression::detail::ExpressionTraits<V>::extents_type>())
  {
    expression::detail::AssignHelper<V, Derived>::assign(v, derived());
  }
};
template <typename Derived_>
inline auto
DenseStorageExpressionBase<Derived_>::const_coeff_ref_linear(index_type i) const
    -> const value_type & {
  return accessor().const_coeff_ref(i);
}

template <typename Derived_>
inline auto DenseStorageExpressionBase<Derived_>::as_mdspan() -> mdspan_type
  requires(!is_const)
{
  if constexpr (IsStatic) {
    return mdspan_type(data());
  } else {
    return mdspan_type(data(), extents());
  }
}

template <typename Derived_>
inline auto DenseStorageExpressionBase<Derived_>::as_mdspan() const
    -> const mdspan_type {
  if constexpr (IsStatic) {
    return mdspan_type(data());
  } else {
    return mdspan_type(data(), extents());
  }
}

template <typename Derived_>
inline auto DenseStorageExpressionBase<Derived_>::as_std_span() const
    -> const_span_type {
  return accessor().as_std_span();
}

template <typename Derived_>
template <typename E2>
inline void DenseStorageExpressionBase<Derived_>::resize(const E2 &e)
  requires(!is_const)
{
  return derived().resize(e);
}

template <typename Derived_>
inline auto DenseStorageExpressionBase<Derived_>::coeff_ref_linear(index_type i)
    -> value_type &
  requires(!is_const)
{
  return accessor().coeff_ref(i);
}

template <typename Derived_>
inline auto
DenseStorageExpressionBase<Derived_>::coeff_linear(index_type i) const
    -> value_type {
  return accessor().coeff(i);
}

template <typename Derived_>
inline auto DenseStorageExpressionBase<Derived_>::accessor() const
    -> const value_accessor_type & {
  return derived().linear_access();
}

template <typename Derived_>
inline auto DenseStorageExpressionBase<Derived_>::data() -> value_type *
  requires(!is_const)
{
  return accessor().data();
}

} // namespace zipper::expression::nullary
#endif
