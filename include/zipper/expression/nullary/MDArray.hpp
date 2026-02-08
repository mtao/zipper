#if !defined(ZIPPER_EXPRESSION_NULLARY_MDARRAY_HPP)
#define ZIPPER_EXPRESSION_NULLARY_MDARRAY_HPP

#include "LinearLayoutExpression.hpp"
#include "MDSpan.hpp"
#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"
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
          Extents, LayoutPolicy, AccessorPolicy,
          MDArray<ElementType, Extents, LayoutPolicy, AccessorPolicy>> {

  using base_type = LinearLayoutExpression<
      storage::DenseData<ElementType,
                         zipper::detail::ExtentsTraits<Extents>::static_size>,
      Extents, LayoutPolicy, AccessorPolicy,
      MDArray<ElementType, Extents, LayoutPolicy, AccessorPolicy>>;
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

  constexpr static bool IsStatic = base_type::extents_traits::is_static;
  using span_type = MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
public:
  using typename base_type::extents_traits;
  using typename base_type::extents_type;
  using base_type::linear_accessor;
  using self_type = MDArray<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

  template <concepts::Expression V>
  void assign(const V &v)
    requires(zipper::utils::extents::assignable_extents_v<
                typename V::extents_type, extents_type>)
  {
    expression::detail::AssignHelper<V, self_type>::assign(v, *this);
  }
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

namespace zipper::expression {
/// ExpressionTraits for MDArray forwards to the LinearLayoutExpression traits,
/// since MDArray is a thin owning wrapper around LinearLayoutExpression.
template <typename ElementType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ExpressionTraits<nullary::MDArray<
    ElementType, Extents, LayoutPolicy, AccessorPolicy>>
    : public detail::ExpressionTraits<nullary::LinearLayoutExpression<
          storage::DenseData<ElementType,
                             zipper::detail::ExtentsTraits<Extents>::static_size>,
          Extents, LayoutPolicy, AccessorPolicy,
          nullary::MDArray<ElementType, Extents, LayoutPolicy, AccessorPolicy>>> {
};
} // namespace zipper::expression
#endif
