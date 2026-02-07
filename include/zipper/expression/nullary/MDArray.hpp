#if !defined(ZIPPER_EXPRESSION_NULLARY_MDARRAY_HPP)
#define ZIPPER_EXPRESSION_NULLARY_MDARRAY_HPP

#include "LinearLayoutExpression.hpp"
#include "MDSpan.hpp"
#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/storage/DenseData.hpp"
#include "zipper/storage/layout_types.hpp"

namespace zipper::expression::nullary {
template <typename ElementType, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<ElementType>>
class MDArray
    : public LinearLayoutExpression<
          storage::DenseData<
              ElementType, zipper::detail::ExtentsTraits<Extents>::static_size>,
          Extents, LayoutPolicy, AccessorPolicy> {

  using base_type = LinearLayoutExpression<
      storage::DenseData<ElementType,
                         zipper::detail::ExtentsTraits<Extents>::static_size>,
      Extents, LayoutPolicy, AccessorPolicy>;
  using base_type::base_type;
  /*
  // defaults are stored in MDSpan
  template <typename ValueType, concepts::Extents Extents, typename
  LayoutPolicy, typename AccessorPolicy> class MDArray : public
  expression::nullary::DenseStorageExpressionBase< MDArray<ValueType, Extents,
  LayoutPolicy, AccessorPolicy>> { public: using self_type = MDArray<ValueType,
  Extents, LayoutPolicy, AccessorPolicy>; using ParentType =
  DenseStorageExpressionBase<self_type>;

    using traits = expression::detail::ExpressionTraits<self_type>;
    using value_type = typename traits::value_type;
    /// Type of the underlying variable (should be seomthing like remove_cvref_t
    using element_type = typename traits::element_type;
    using extents_type = Extents;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;


  */
  using extents_traits = typename base_type::extents_traits;
  using extents_type = typename base_type::extents_type;
  constexpr static bool IsStatic = extents_traits::is_static;
  using span_type = MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
  using base_type::linear_accessor;
  /*
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

  */
  auto as_span() -> span_type {
    if constexpr (IsStatic) {
      return span_type(linear_accessor().as_std_span());
    } else {
      const extents_type &e = extents();
      return span_type(linear_accessor().as_std_span(), e);
    }
  }
  auto as_span() const -> const span_type {
    if constexpr (IsStatic) {
      return span_type(linear_accessor().as_std_span());
    } else {
      const extents_type &e = extents();
      return span_type(linear_accessor().as_std_span(), e);
    }
  }

  /*
    template <typename... Args>
    MDArray(const extents_type &extents, Args &&...args)
        : ParentType(extents), m_accessor(std::forward<Args>(args)...) {}

  */
  template <zipper::concepts::Extents E2>
  void resize(const E2 &e)
    requires(extents_traits::template is_convertable_from<E2>() && !IsStatic)
  {
    static_assert(E2::rank() != 0);
    this->resize_extents(e);
    linear_accessor().container().resize(
        zipper::detail::ExtentsTraits<E2>::size(e));
  }
  /*
    using iterator_type = accessor_type::iterator_type;
    using const_iterator_type = accessor_type::const_iterator_type;
    auto begin() { return m_accessor.begin(); }
    auto end() { return m_accessor.end(); }
    auto begin() const { return m_accessor.begin(); }
    auto end() const { return m_accessor.end(); }

  private:
    accessor_type m_accessor;
    */
};

} // namespace zipper::expression::nullary
/*
namespace zipper::expression {

template <typename ValueType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ExpressionTraits<zipper::expression::nullary::MDArray<
    ValueType, Extents, LayoutPolicy, AccessorPolicy>>
    : public detail::DefaultExpressionTraits<ValueType, Extents>
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
} // namespace zipper::expression::nullary
*/
#endif
