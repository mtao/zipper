#pragma once
#include "zipper/expression/ExpressionBase.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"
#include "zipper/storage/StlStorageInfo.hpp"
namespace zipper::expression::nullary {

template <typename S, typename Policy = storage::NoUnwrap>
struct StlMDArray : public ExpressionBase<StlMDArray<S, Policy>> {
public:
  using traits = expression::detail::ExpressionTraits<StlMDArray<S, Policy>>;
  using info_helper = storage::StlStorageInfo<std::decay_t<S>, Policy>;
  using extents_type = typename info_helper::extents_type;
  using value_type = typename info_helper::value_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  constexpr static bool IsStatic = extents_traits::is_static;
  consteval static auto rank() -> index_type { return info_helper::rank(); }
  consteval static auto static_extent(rank_type) -> index_type {
    return extents_type::static_extent;
  }

  auto extents() const -> extents_type { return info_helper::make_extents(m_data); }
  auto extent(rank_type i) const -> index_type { return extents().extent(i); }

  StlMDArray()
    requires(!IsStatic)
      : m_data(info_helper::initialize(extents_type{}, value_type{0.0})) {}

  explicit StlMDArray(const extents_type &e, const value_type &default_value = 0.0)
    requires(!IsStatic)
      : m_data(info_helper::initialize(e, default_value)) {}

  StlMDArray()
    requires(IsStatic)
      : m_data(info_helper::initialize(value_type{0.0})) {}

  explicit StlMDArray(const value_type &default_value)
    requires(IsStatic)
      : m_data(info_helper::initialize(default_value)) {}

  explicit StlMDArray(S &d) : m_data(d) {}
  explicit StlMDArray(S &&d)
    requires(!std::is_reference_v<S>)
      : m_data(std::move(d)) {}

  void resize(const extents_type &e) {
    info_helper::resize(m_data, e);
  }

  template <zipper::concepts::Index... Args>
  auto const_coeff_ref(Args &&...args) const -> const value_type & {
    return info_helper::get_value(m_data, index_type(args)...);
  }
  template <zipper::concepts::Index... Args>
  auto coeff_ref(Args &&...args) -> value_type &
    requires traits::is_writable
  {
    return info_helper::get_value(m_data, index_type(args)...);
  }
  template <zipper::concepts::Index... Args>
  auto coeff(Args &&...args) const -> value_type {
    return info_helper::get_value(m_data, index_type(args)...);
  }

  template <zipper::concepts::Expression V>
  void assign(const V &v)
    requires(extents_traits::template is_convertable_from<
             typename expression::detail::ExpressionTraits<V>::extents_type>())
  {
    expression::detail::AssignHelper<V, StlMDArray<S, Policy>>::assign(v, *this);
  }

private:
  S m_data;
};

// Deduction guides (default policy)
template <typename S> StlMDArray(const S &) -> StlMDArray<S>;
template <typename S> StlMDArray(S &&) -> StlMDArray<S>;

// Non-owning view helpers (default policy — no unwrapping)
template <typename S> auto get_non_owning_stl_storage(const S &s) {
  return StlMDArray<const S &>(s);
}
template <typename S> auto get_const_non_owning_stl_storage(const S &s) {
  return StlMDArray<const S &>(s);
}
template <typename S> auto get_non_owning_stl_storage(S &s) {
  return StlMDArray<S &>(s);
}

// Non-owning view helpers with explicit policy
template <typename Policy, typename S> auto get_non_owning_stl_storage(const S &s) {
  return StlMDArray<const S &, Policy>(s);
}
template <typename Policy, typename S> auto get_const_non_owning_stl_storage(const S &s) {
  return StlMDArray<const S &, Policy>(s);
}
template <typename Policy, typename S> auto get_non_owning_stl_storage(S &s) {
  return StlMDArray<S &, Policy>(s);
}

} // namespace zipper::expression::nullary
namespace zipper::expression {

// ExpressionTraits for StlMDArray with policy support
template <typename S, typename Policy>
struct detail::ExpressionTraits<zipper::expression::nullary::StlMDArray<S, Policy>>
    : public detail::BasicExpressionTraits<
          typename zipper::storage::StlStorageInfo<std::decay_t<S>, Policy>::value_type,
          typename zipper::storage::StlStorageInfo<
              std::decay_t<S>, Policy>::extents_type,
          zipper::detail::AccessFeatures{
              .is_const = std::is_const_v<std::remove_reference_t<S>>,
              .is_reference = true},
          zipper::detail::ShapeFeatures{
              .is_resizable =
                  (zipper::storage::StlStorageInfo<
                      std::decay_t<S>, Policy>::extents_type::rank_dynamic() > 0)}> {
  constexpr static bool is_coefficient_consistent = true;
  /// StlMDArray stores references when S is a reference type (borrowing mode).
  constexpr static bool stores_references = std::is_reference_v<S>;
};
} // namespace zipper::expression
