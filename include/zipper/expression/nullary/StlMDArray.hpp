#include "zipper/expression/SizedExpressionBase.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"
#include "zipper/storage/StlStorageInfo.hpp"
namespace zipper::expression::nullary {

template <typename S>
struct StlMDArray : public SizedExpressionBase<StlMDArray<S>> {
public:
  using Base = zipper::expression::SizedExpressionBase<StlMDArray<S>>;
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

  StlMDArray(const extents_type &e = {}, const value_type &default_value = 0.0)
    requires(!IsStatic)
      : Base(e), m_data(info_helper::initialize(e, default_value)) {}

  StlMDArray(const value_type &default_value = 0.0)
    requires(IsStatic)
      : Base(), m_data(info_helper::initialize(default_value)) {}
  StlMDArray(S &d) : Base(info_helper::make_extents(d)), m_data(d) {}

  void resize(const extents_type &e) {
    Base::set_extent(e);
    info_helper::resize(m_data, e);
  }

  // extents_type extents() const { return info_helper::make_extents(m_data);
  // }

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
              std::decay_t<S>>::extents_type>
/*: public detail::ExpressionTraits <
  views::StorageExpressionBase<zipper::storage::SpanStorage<
      ValueType, Extents, LayoutPolicy, AccessorPolicy>> */
{
  using StlType = std::decay_t<S>;
  using value_type =
      typename zipper::storage::StlStorageInfo<StlType>::value_type;
  constexpr static bool is_const = std::is_const_v<S>;
  using extents_type =
      typename zipper::storage::StlStorageInfo<StlType>::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  constexpr static bool is_writable = !is_const;
  constexpr static bool is_coefficient_consistent = true;
  constexpr static bool is_resizable = extents_type::rank_dynamic() > 0;
};
} // namespace zipper::expression
