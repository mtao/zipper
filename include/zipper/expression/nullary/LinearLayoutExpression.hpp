#if !defined(ZIPPER_EXPRESSION_NULLARY_LINEARLAYOUTEXPRESSION_HPP)
#define ZIPPER_EXPRESSION_NULLARY_LINEARLAYOUTEXPRESSION_HPP

#include "zipper/expression/ExpressionBase.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/storage/LinearAccessorTraits.hpp"
#include "zipper/storage/layout_types.hpp"

namespace zipper::expression {
namespace nullary {

namespace _detail_lle {
/// Helper to resolve the CRTP derived type: if FinalDerived is void,
/// the class uses itself; otherwise it forwards FinalDerived to ExpressionBase.
template <typename FinalDerived, typename Self>
using resolved_derived_t = std::conditional_t<std::is_void_v<FinalDerived>, Self, FinalDerived>;
} // namespace _detail_lle

/// Expression whose values are held in a linear span with its mapping type.
///
/// @tparam FinalDerived  The outermost CRTP derived type (e.g. MDArray).
///                       Pass `void` (the default) to use LinearLayoutExpression
///                       itself as the CRTP leaf.
template <typename LinearAccessorType, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<
              typename LinearAccessorType::element_type>,
          typename FinalDerived = void>
class LinearLayoutExpression
    : public ExpressionBase<_detail_lle::resolved_derived_t<
          FinalDerived,
          LinearLayoutExpression<LinearAccessorType, Extents, LayoutPolicy,
                                 AccessorPolicy, FinalDerived>>> {
public:
  using self_type = LinearLayoutExpression<LinearAccessorType, Extents,
                                           LayoutPolicy, AccessorPolicy,
                                           FinalDerived>;
  using traits = expression::detail::ExpressionTraits<self_type>;
  using value_type = typename traits::value_type;
  using element_type = typename traits::element_type;
  using extents_type = traits::extents_type;

  using layout_policy = LayoutPolicy;

  using mapping_type = typename layout_policy::template mapping<extents_type>;
  using accessor_policy = AccessorPolicy;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

  using linear_accessor_type = LinearAccessorType;
  using span_type = std::span<value_type, linear_accessor_type::static_size>;
  using const_span_type =
      std::span<const value_type, linear_accessor_type::static_size>;
  using mdspan_type =
      zipper::mdspan<value_type, extents_type, layout_policy, accessor_policy>;
  using const_mdspan_type = zipper::mdspan<const value_type, extents_type,
                                           layout_policy, accessor_policy>;

  constexpr auto mapping() const -> const mapping_type & { return m_mapping; }
  constexpr auto extents() const -> const extents_type & {
    return mapping().extents();
  }
  [[nodiscard]] constexpr auto rank() const -> rank_type {
    return extents().rank();
  }
  [[nodiscard]] constexpr auto extent(index_type i) const -> index_type {
    return extents().extent(i);
  }

  LinearLayoutExpression(const LinearLayoutExpression &) = default;
  LinearLayoutExpression(LinearLayoutExpression &&) = default;
  auto operator=(const LinearLayoutExpression &)
      -> LinearLayoutExpression & = default;
  auto operator=(LinearLayoutExpression &&)
      -> LinearLayoutExpression & = default;

  LinearLayoutExpression(const extents_type &extents = {})
    requires(std::is_default_constructible_v<linear_accessor_type>)
      : m_linear_accessor(), m_mapping(extents) {
    if constexpr (!extents_traits::is_static) {
      m_linear_accessor = linear_accessor_type(extents_traits::size(extents));
    }
  }

  // Constructor just forwards everything to the linear accessor. If a dynamic
  // sized attribute is used then linear access must come with extents

  LinearLayoutExpression(const linear_accessor_type &linear_access,
                         const extents_type &extents)
      : m_linear_accessor(linear_access), m_mapping(extents) {}

  /// Constructor for static extents might want to skip putting an extents
  /// member
  LinearLayoutExpression(linear_accessor_type const &linear_access)
    requires(extents_traits::is_static)
      : LinearLayoutExpression(linear_access, extents_type{}) {}

  /// Converting constructor: allows e.g. LinearLayoutExpression<SpanData<T,...>,...>
  /// → LinearLayoutExpression<SpanData<const T,...>,...> when the linear
  /// accessor is convertible.
  template <typename OtherLinearAccessor, typename OtherFinalDerived>
  LinearLayoutExpression(
      const LinearLayoutExpression<OtherLinearAccessor, Extents, LayoutPolicy,
                                   AccessorPolicy, OtherFinalDerived> &other)
    requires(std::is_constructible_v<LinearAccessorType,
                                     const OtherLinearAccessor &> &&
             !std::is_same_v<LinearAccessorType, OtherLinearAccessor>)
      : m_linear_accessor(other.linear_accessor()), m_mapping(other.mapping()) {}

  template <concepts::Extents E2>
  void resize_extents(const E2 &e)
    requires(extents_traits::template is_convertable_from<E2>())
  {
    m_mapping = mapping_type(extents_traits::convert_from(e));
  }

  auto linear_accessor() -> linear_accessor_type & { return m_linear_accessor; }
  auto linear_accessor() const -> const linear_accessor_type & {
    return m_linear_accessor;
  }
  auto data() -> value_type *
    requires(!traits::is_const_valued())
  {
    return linear_accessor().data();
  }
  auto data() const -> const value_type * { return linear_accessor().data(); }

  auto as_mdspan() -> mdspan_type
    requires(!traits::is_const_valued());
  auto as_mdspan() const -> const_mdspan_type;

  // todo fixing names
  auto as_std_span() -> span_type { return linear_accessor().as_std_span(); }
  auto as_std_span() const -> const_span_type {
    return linear_accessor().as_std_span();
  }

protected:
  template <concepts::IndexArgument... Indices>
  auto get_index(Indices &&...indices) const -> index_type {
    // rank-0 expressions act as scalars; any indices are ignored
    if constexpr (extents_type::rank() == 0) {
      return 0;
    } else {
      static_assert((std::is_integral_v<std::decay_t<Indices>> && ...));
#if !defined(NDEBUG)
      assert(zipper::utils::extents::indices_in_range(extents(), indices...));
#endif
      return mapping()(std::forward<Indices>(indices)...);
    }
  }

public:
  template <concepts::Index... Indices>
  auto coeff(Indices &&...indices) const -> value_type {
    index_type idx = get_index(std::forward<Indices>(indices)...);
    return m_linear_accessor.coeff(idx);
  }
  template <concepts::Index... Indices>
  auto coeff_ref(Indices &&...indices) -> value_type &
    requires(traits::is_assignable())
  {
    index_type idx = get_index(std::forward<Indices>(indices)...);
    return m_linear_accessor.coeff_ref(idx);
  }

  template <concepts::Index... Indices>
  auto const_coeff_ref(Indices &&...indices) const -> const value_type &
    requires(traits::is_referrable())
  {
    index_type idx = get_index(std::forward<Indices>(indices)...);
    return m_linear_accessor.const_coeff_ref(idx);
  }

private:
  LinearAccessorType m_linear_accessor;
  mapping_type m_mapping;
};

template <typename LinearAccessorType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy, typename FinalDerived>
auto LinearLayoutExpression<LinearAccessorType, Extents, LayoutPolicy,
                            AccessorPolicy, FinalDerived>::as_mdspan() -> mdspan_type
  requires(!traits::is_const_valued())
{
  return mdspan_type(data(), extents());
}
template <typename LinearAccessorType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy, typename FinalDerived>
auto LinearLayoutExpression<LinearAccessorType, Extents, LayoutPolicy,
                            AccessorPolicy, FinalDerived>::as_mdspan() const
    -> const_mdspan_type {
  return const_mdspan_type(data(), extents());
}

} // namespace nullary
template <typename LinearAccessorType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy, typename FinalDerived>
struct detail::ExpressionTraits<nullary::LinearLayoutExpression<
    LinearAccessorType, Extents, LayoutPolicy, AccessorPolicy, FinalDerived>>
    : public BasicExpressionTraits<
          typename LinearAccessorType::value_type, Extents,
          storage::template LinearAccessorTraits<
              LinearAccessorType>::access_features,
          storage::LinearAccessorTraits<LinearAccessorType>::shape_features> {

  using BaseTraits = BasicExpressionTraits<
      typename LinearAccessorType::value_type, Extents,
      storage::LinearAccessorTraits<LinearAccessorType>::access_features,
      storage::LinearAccessorTraits<LinearAccessorType>::shape_features>;

  using extents_traits = BaseTraits::extents_traits;
  using extents_type = BaseTraits::extents_type;
  using layout_policy = LayoutPolicy;
  using accessor_policy = AccessorPolicy;
  using mapping_type = typename layout_policy::template mapping<extents_type>;
};
} // namespace zipper::expression
#endif
