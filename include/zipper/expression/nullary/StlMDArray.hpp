#include "zipper/expression/ExpressionBase.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"
#include "zipper/storage/StlStorageInfo.hpp"
namespace zipper::expression::nullary {

template <typename S>
struct StlMDArray : public ExpressionBase<StlMDArray<S>> {
public:
  using traits = expression::detail::ExpressionTraits<StlMDArray<S>>;
  using info_helper = storage::StlStorageInfo<std::decay_t<S>>;
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

  StlMDArray(const extents_type &e = {}, const value_type &default_value = 0.0)
    requires(!IsStatic)
      : m_data(info_helper::initialize(e, default_value)) {}

  StlMDArray(const value_type &default_value = 0.0)
    requires(IsStatic)
      : m_data(info_helper::initialize(default_value)) {}
  StlMDArray(S &d) : m_data(d) {}

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
    expression::detail::AssignHelper<V, StlMDArray<S>>::assign(v, *this);
  }

private:
  S m_data;
};

template <typename S> StlMDArray(const S &) -> StlMDArray<S>;

template <typename S> auto get_non_owning_stl_storage(const S &s) {
  return StlMDArray<const S &>(s);
}
template <typename S> auto get_const_non_owning_stl_storage(const S &s) {
  return StlMDArray<const S &>(s);
}
template <typename S> auto get_non_owning_stl_storage(S &s) {
  return StlMDArray<S &>(s);
}
} // namespace zipper::expression::nullary
namespace zipper::expression {
template <typename S>
struct detail::ExpressionTraits<zipper::expression::nullary::StlMDArray<S>>
    : public detail::DefaultExpressionTraits<
          typename zipper::storage::StlStorageInfo<std::decay_t<S>>::value_type,
          typename zipper::storage::StlStorageInfo<
              std::decay_t<S>>::extents_type> {
  using StlType = std::decay_t<S>;
  using value_type =
      typename zipper::storage::StlStorageInfo<StlType>::value_type;
  using extents_type =
      typename zipper::storage::StlStorageInfo<StlType>::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

  constexpr static AccessFeatures access_features = {
      .is_const = std::is_const_v<std::remove_reference_t<S>>,
      .is_reference = true,
      .is_alias_free = true,
  };
  constexpr static ShapeFeatures shape_features = {
      .is_resizable = extents_type::rank_dynamic() > 0,
  };

  consteval static auto is_const_valued() -> bool {
    return access_features.is_const;
  }
  consteval static auto is_reference_valued() -> bool {
    return access_features.is_reference;
  }
  consteval static auto is_assignable() -> bool {
    return access_features.is_assignable();
  }
  consteval static auto is_referrable() -> bool {
    return access_features.is_reference;
  }
  consteval static auto is_resizable() -> bool {
    return shape_features.is_resizable;
  }

  constexpr static bool is_writable = is_assignable();
};
} // namespace zipper::expression
