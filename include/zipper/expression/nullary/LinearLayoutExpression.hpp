#if !defined(ZIPPER_EXPRESSION_NULLARY_LINEARLAYOUTEXPRESSION_HPP)
#define ZIPPER_EXPRESSION_NULLARY_LINEARLAYOUTEXPRESSION_HPP

#include "zipper/detail/assert.hpp"
#include "zipper/detail/no_unique_address.hpp"
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
using resolved_derived_t =
    std::conditional_t<std::is_void_v<FinalDerived>, Self, FinalDerived>;
} // namespace _detail_lle

/// Layout for owning storage that copies a view with LayoutPolicy: the same
/// layout when its mapping is determined by the extents alone, otherwise
/// (e.g. layout_stride, whose strides are run-time) the default layout.
template <typename LayoutPolicy, typename Extents>
using owning_layout_policy_t = std::conditional_t<
    std::is_constructible_v<
        typename LayoutPolicy::template mapping<Extents>, const Extents &>,
    LayoutPolicy, default_layout_policy>;

/// Expression whose values are held in a linear span with its mapping type.
///
/// @tparam FinalDerived  The outermost CRTP derived type (e.g. MDArray).
///                       Pass `void` (the default) to use
///                       LinearLayoutExpression itself as the CRTP leaf.
template <typename LinearAccessorType, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename FinalDerived = void>
class LinearLayoutExpression
    : public ExpressionBase<_detail_lle::resolved_derived_t<
          FinalDerived,
          LinearLayoutExpression<LinearAccessorType, Extents, LayoutPolicy,
                                 FinalDerived>>> {
public:
  using self_type =
      LinearLayoutExpression<LinearAccessorType, Extents, LayoutPolicy,
                             FinalDerived>;
  using traits = expression::detail::ExpressionTraits<self_type>;
  using value_type = typename traits::value_type;
  using element_type = typename traits::element_type;
  using extents_type = traits::extents_type;

  using layout_policy = LayoutPolicy;

  using mapping_type = typename layout_policy::template mapping<extents_type>;
  /// The mapping is determined by the extents alone (false for e.g.
  /// layout_stride); otherwise it must be supplied on construction.
  constexpr static bool mapping_from_extents =
      std::is_constructible_v<mapping_type, const extents_type &>;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

  using linear_accessor_type = LinearAccessorType;
  /// Elements live in a raw buffer: data(), operator[], as_std_span() and
  /// as_mdspan() are available.
  constexpr static bool is_contiguous =
      storage::LinearAccessorTraits<linear_accessor_type>::is_contiguous;
  using span_type = std::span<value_type, linear_accessor_type::static_size>;
  using const_span_type =
      std::span<const value_type, linear_accessor_type::static_size>;

  /// Whether the linear accessor supports zipper::uninitialized
  /// construction for these extents.
  constexpr static bool uninitialized_constructible = [] {
    if constexpr (extents_traits::is_static) {
      return std::is_constructible_v<linear_accessor_type, uninitialized_t>;
    } else {
      return std::is_constructible_v<linear_accessor_type, uninitialized_t,
                                     index_type>;
    }
  }();

  constexpr auto mapping() const -> const mapping_type & { return m_mapping; }
  constexpr auto extents() const -> extents_type { return mapping().extents(); }
  [[nodiscard]] static constexpr auto rank() -> rank_type {
    return extents_type::rank();
  }
  [[nodiscard]] constexpr auto extent(index_type i) const -> index_type {
    return m_mapping.extents().extent(i);
  }

  LinearLayoutExpression(const LinearLayoutExpression &) = default;
  LinearLayoutExpression(LinearLayoutExpression &&) = default;
  auto operator=(const LinearLayoutExpression &)
      -> LinearLayoutExpression & = default;
  auto operator=(LinearLayoutExpression &&)
      -> LinearLayoutExpression & = default;

  LinearLayoutExpression()
    requires(std::is_default_constructible_v<linear_accessor_type> &&
             mapping_from_extents)
      : LinearLayoutExpression(extents_type{}) {}

  explicit LinearLayoutExpression(const extents_type &extents)
    requires(std::is_default_constructible_v<linear_accessor_type> &&
             mapping_from_extents)
      : m_linear_accessor(make_linear_accessor(extents)), m_mapping(extents) {}

  /// Allocates storage without initializing its elements. See
  /// zipper::uninitialized_t.
  explicit LinearLayoutExpression(uninitialized_t)
    requires(uninitialized_constructible && mapping_from_extents)
      : LinearLayoutExpression(uninitialized, extents_type{}) {}

  LinearLayoutExpression(uninitialized_t, const extents_type &extents)
    requires(uninitialized_constructible && mapping_from_extents)
      : m_linear_accessor(make_linear_accessor(uninitialized, extents)),
        m_mapping(extents) {}

  // Constructor just forwards everything to the linear accessor. If a dynamic
  // sized attribute is used then linear access must come with extents

  LinearLayoutExpression(const linear_accessor_type &linear_access,
                         const extents_type &extents)
    requires(mapping_from_extents)
      : m_linear_accessor(linear_access), m_mapping(extents) {}

  /// Explicit mapping, e.g. the strides of a layout_stride view.
  LinearLayoutExpression(const linear_accessor_type &linear_access,
                         const mapping_type &mapping)
      : m_linear_accessor(linear_access), m_mapping(mapping) {}

  /// Constructor for static extents might want to skip putting an extents
  /// member
  LinearLayoutExpression(linear_accessor_type const &linear_access)
    requires(extents_traits::is_static && mapping_from_extents)
      : LinearLayoutExpression(linear_access, extents_type{}) {}

  /// Converting constructor: allows e.g.
  /// LinearLayoutExpression<SpanData<T,...>,...> →
  /// LinearLayoutExpression<SpanData<const T,...>,...> when the linear accessor
  /// is convertible.
  template <typename OtherLinearAccessor, typename OtherFinalDerived>
  LinearLayoutExpression(
      const LinearLayoutExpression<OtherLinearAccessor, Extents, LayoutPolicy,
                                   OtherFinalDerived> &other)
    requires(std::is_constructible_v<LinearAccessorType,
                                     const OtherLinearAccessor &> &&
             !std::is_same_v<LinearAccessorType, OtherLinearAccessor>)
      : m_linear_accessor(other.linear_accessor()), m_mapping(other.mapping()) {
  }

  template <zipper::concepts::Extents E2>
  void resize_extents(const E2 &e)
    requires(extents_traits::template is_convertable_from<E2>() &&
             mapping_from_extents)
  {
    m_mapping = mapping_type(extents_traits::convert_from(e));
  }

  auto linear_accessor() -> linear_accessor_type & { return m_linear_accessor; }
  auto linear_accessor() const -> const linear_accessor_type & {
    return m_linear_accessor;
  }
  auto data() -> value_type *
    requires(is_contiguous && !traits::is_const_valued())
  {
    return linear_accessor().data();
  }
  auto data() const -> const value_type *
    requires(is_contiguous)
  {
    return linear_accessor().data();
  }

  // Flat unchecked linear access into the underlying buffer (the `is_linear_array`
  // contract: `(*this)[mapping()(i,j,...)] == (*this)(i,j,...)`). Deliberately
  // unchecked — bounds-checked indexing (e.g. std::span's under
  // _GLIBCXX_ASSERTIONS) inserts branches that defeat vectorization in hot
  // kernels.
  constexpr auto operator[](index_type k) -> value_type &
    requires(is_contiguous && !traits::is_const_valued())
  {
    return linear_accessor().data()[k];
  }
  constexpr auto operator[](index_type k) const -> const value_type &
    requires(is_contiguous)
  {
    return linear_accessor().data()[k];
  }

  /// View the raw storage as an mdspan.  The accessor policy is chosen by
  /// the caller; it must operate on a raw pointer to the (const) element
  /// type, e.g. default_accessor (the default), aligned_accessor or
  /// scaled_accessor.
  template <typename Accessor = default_accessor_policy<value_type>>
    requires(is_contiguous && !traits::is_const_valued() &&
             std::is_same_v<typename Accessor::element_type, value_type> &&
             std::is_convertible_v<value_type *,
                                   typename Accessor::data_handle_type>)
  auto as_mdspan(const Accessor &accessor = {})
      -> zipper::mdspan<value_type, extents_type, layout_policy, Accessor> {
    return {data(), mapping(), accessor};
  }
  template <typename Accessor = default_accessor_policy<const value_type>>
    requires(is_contiguous &&
             std::is_same_v<typename Accessor::element_type,
                            const value_type> &&
             std::is_convertible_v<const value_type *,
                                   typename Accessor::data_handle_type>)
  auto as_mdspan(const Accessor &accessor = {}) const
      -> zipper::mdspan<const value_type, extents_type, layout_policy,
                        Accessor> {
    return {data(), mapping(), accessor};
  }

  // todo fixing names
  auto as_std_span() -> span_type
    requires(is_contiguous)
  {
    return linear_accessor().as_std_span();
  }
  auto as_std_span() const -> const_span_type
    requires(is_contiguous)
  {
    return linear_accessor().as_std_span();
  }

protected:
  template <zipper::concepts::IndexArgument... Indices>
  auto get_index(Indices &&...indices) const -> index_type {
    // rank-0 expressions act as scalars; any indices are ignored
    if constexpr (extents_type::rank() == 0) {
      return 0;
    } else {
      static_assert((std::is_integral_v<std::decay_t<Indices>> && ...));
      ZIPPER_ASSERT(
          zipper::utils::extents::indices_in_range(extents(), indices...));
      return mapping()(std::forward<Indices>(indices)...);
    }
  }

public:
  template <zipper::concepts::Index... Indices>
  auto coeff(Indices &&...indices) const -> value_type {
    index_type idx = get_index(std::forward<Indices>(indices)...);
    return m_linear_accessor.coeff(idx);
  }
  template <zipper::concepts::Index... Indices>
  auto coeff_ref(Indices &&...indices) -> value_type &
    requires(traits::is_assignable())
  {
    index_type idx = get_index(std::forward<Indices>(indices)...);
    return m_linear_accessor.coeff_ref(idx);
  }

  template <zipper::concepts::Index... Indices>
  auto const_coeff_ref(Indices &&...indices) const -> const value_type &
    requires(traits::is_referrable())
  {
    index_type idx = get_index(std::forward<Indices>(indices)...);
    return m_linear_accessor.const_coeff_ref(idx);
  }

private:
  static auto make_linear_accessor(const extents_type &extents)
      -> linear_accessor_type {
    if constexpr (extents_traits::is_static) {
      return linear_accessor_type{};
    } else {
      return linear_accessor_type(extents_traits::size(extents));
    }
  }
  static auto make_linear_accessor(uninitialized_t,
                                   const extents_type &extents)
      -> linear_accessor_type {
    if constexpr (extents_traits::is_static) {
      return linear_accessor_type(uninitialized);
    } else {
      return linear_accessor_type(uninitialized,
                                  extents_traits::size(extents));
    }
  }

  LinearAccessorType m_linear_accessor;
  ZIPPER_NO_UNIQUE_ADDRESS mapping_type m_mapping;
};

} // namespace nullary
template <typename LinearAccessorType, typename Extents, typename LayoutPolicy,
          typename FinalDerived>
struct detail::ExpressionTraits<nullary::LinearLayoutExpression<
    LinearAccessorType, Extents, LayoutPolicy, FinalDerived>>
    : public BasicExpressionTraits<
          typename LinearAccessorType::value_type, Extents,
          storage::template LinearAccessorTraits<
              LinearAccessorType>::access_features,
          storage::LinearAccessorTraits<LinearAccessorType>::shape_features> {
  /// Dense leaf → prefer the layout it was created with.
  using preferred_layout = zipper::detail::DenseLayoutPreference<
      nullary::owning_layout_policy_t<LayoutPolicy, Extents>>;

  /// Linear storage exposes `mapping()`; it is a linear array (flat
  /// `operator[]` / `data()`) only when the storage is a raw buffer.
  constexpr static bool has_layout_mapping = true;
  constexpr static bool is_linear_array =
      storage::LinearAccessorTraits<LinearAccessorType>::is_contiguous;

};
} // namespace zipper::expression
#endif
