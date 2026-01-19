#if !defined(ZIPPER_EXPRESSION_NULLARY_MDARRAY_HPP)
#define ZIPPER_EXPRESSION_NULLARY_MDARRAY_HPP

#include "MDSpan.hpp"
#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/expression/nullary/DenseStorageExpressionBase.hpp"
#include "zipper/storage/DenseData.hpp"
#include "zipper/storage/layout_types.hpp"

namespace zipper::expression::nullary {
// defaults are stored in MDSpan
template <typename ValueType, concepts::Extents Extents, typename LayoutPolicy,
          typename AccessorPolicy>
class MDArray : public expression::nullary::DenseStorageExpressionBase<
                    MDArray<ValueType, Extents, LayoutPolicy, AccessorPolicy>> {
public:
  using self_type = MDArray<ValueType, Extents, LayoutPolicy, AccessorPolicy>;
  using ParentType = DenseStorageExpressionBase<self_type>;

  using traits = expression::detail::ExpressionTraits<self_type>;
  using value_type = typename traits::value_type;
  using element_type = typename traits::element_type;
  using extents_type = Extents;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

  constexpr static bool IsStatic = extents_traits::is_static;

  using span_type = MDSpan<ValueType, Extents, LayoutPolicy, AccessorPolicy>;
  MDArray() : ParentType(), m_accessor() {}

  using accessor_type =
      storage::DenseData<value_type, extents_traits::static_size>;
  auto accessor() const -> const accessor_type & { return m_accessor; }
  auto accessor() -> accessor_type & { return m_accessor; }
  auto linear_access() const -> const accessor_type & { return m_accessor; }
  auto linear_access() -> accessor_type & { return m_accessor; }
  using ParentType::assign;

  MDArray(const MDArray &) = default;
  MDArray(MDArray &&) = default;
  auto operator=(const MDArray &) -> MDArray & = default;
  auto operator=(MDArray &&) -> MDArray & = default;

  MDArray(const extents_type &extents)
    requires(!IsStatic)
      : ParentType(extents), m_accessor(extents_traits::size(extents)) {}
  MDArray(const extents_type &extents)
    requires(IsStatic)
      : ParentType(extents), m_accessor() {}

  auto as_span() -> span_type {
    if constexpr (IsStatic) {
      return span_type(accessor().as_std_span());
    } else {
      const extents_type &e = ParentType::extents();
      return span_type(accessor().as_std_span(), e);
    }
  }
  auto as_span() const -> const span_type {
    if constexpr (IsStatic) {
      return span_type(accessor().as_std_span());
    } else {
      const extents_type &e = ParentType::extents();
      return span_type(accessor().as_std_span(), e);
    }
  }

  template <typename... Args>
  MDArray(const extents_type &extents, Args &&...args)
      : ParentType(extents), m_accessor(std::forward<Args>(args)...) {}

  template <zipper::concepts::Extents E2>
  void resize(const E2 &e)
    requires(extents_traits::template is_convertable_from<E2>() && !IsStatic)
  {
    static_assert(E2::rank() != 0);
    this->resize_extents(e);
    m_accessor.container().resize(zipper::detail::ExtentsTraits<E2>::size(e));
  }
  using iterator_type = accessor_type::iterator_type;
  using const_iterator_type = accessor_type::const_iterator_type;
  auto begin() { return m_accessor.begin(); }
  auto end() { return m_accessor.end(); }
  auto begin() const { return m_accessor.begin(); }
  auto end() const { return m_accessor.end(); }

private:
  accessor_type m_accessor;
};

} // namespace zipper::expression::nullary
namespace zipper::expression {

template <typename ValueType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ExpressionTraits<zipper::expression::nullary::MDArray<
    ValueType, Extents, LayoutPolicy, AccessorPolicy>>
    : public detail::DefaultExpressionTraits<ValueType, Extents>
/*: public detail::ExpressionTraits <
  expression::StorageExpressionBase<zipper::storage::MDArray<
      ValueType, Extents, LayoutPolicy, AccessorPolicy>> */
{
  using value_type = ValueType;
  using extents_type = Extents;
  using extents_traits = typename zipper::detail::ExtentsTraits<extents_type>;
  using value_accessor_type =
      storage::DenseData<ValueType, extents_traits::static_size>;
  using layout_policy = LayoutPolicy;
  using accessor_policy = AccessorPolicy;
  using mapping_type = typename layout_policy::template mapping<extents_type>;
  constexpr static bool is_const = false;
  constexpr static bool is_writable = true;
  constexpr static bool is_coefficient_consistent = true;
  constexpr static bool is_resizable = extents_type::rank_dynamic() > 0;
};
} // namespace zipper::expression
#endif
