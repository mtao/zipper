#if !defined(ZIPPER_EXPRESSION_NULLARY_MDSPAN_HPP)
#define ZIPPER_EXPRESSION_NULLARY_MDSPAN_HPP

#include "DenseStorageExpressionBase.hpp"
#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/storage/SpanData.hpp"
#include "zipper/storage/layout_types.hpp"

namespace zipper::expression::nullary {
template <typename ElementType, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<ElementType>>
class PlainObjectStorage;

template <typename ElementType, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<ElementType>>
class MDSpan : public DenseStorageExpressionBase<
                   MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>> {
public:
  using self_type = MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
  using ParentType = DenseStorageExpressionBase<self_type>;
  using traits = expression::detail::ExpressionTraits<self_type>;
  using value_type = typename traits::value_type;
  using element_type = typename traits::element_type;
  using extents_type = Extents;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

  using std_span_type = std::span<ElementType, extents_traits::static_size>;
  constexpr static bool IsStatic =
      zipper::detail::ExtentsTraits<extents_type>::is_static;
  MDSpan()
    requires(IsStatic)
      : ParentType(), m_accessor() {}

  using accessor_type =
      storage::SpanData<ElementType, extents_traits::static_size>;
  auto accessor() const -> const accessor_type & { return m_accessor; }
  auto accessor() -> accessor_type & { return m_accessor; }
  auto linear_access() const -> const accessor_type & { return m_accessor; }
  auto linear_access() -> accessor_type & { return m_accessor; }
  using ParentType::assign;

  MDSpan(const MDSpan &) = default;
  MDSpan(MDSpan &&) = default;
  auto operator=(const MDSpan &) -> MDSpan & = delete;
  auto operator=(MDSpan &&) -> MDSpan & = delete;

  MDSpan(const std_span_type &s, const extents_type &extents)
    requires(!IsStatic)
      : ParentType(extents), m_accessor(s) {}

  MDSpan(const std_span_type &s)
    requires(IsStatic)
      : ParentType(), m_accessor(s) {}

  MDSpan(const std::span<value_type, extents_traits::static_size> &s,
         const extents_type &extents)
    requires(!IsStatic && !std::same_as<element_type, value_type>)
      : ParentType(extents), m_accessor(s) {}

  MDSpan(const std::span<value_type, extents_traits::static_size> &s)
    requires(IsStatic && !std::same_as<element_type, value_type>)
      : ParentType(), m_accessor(s) {}

  template <typename AP>
  MDSpan(const MDSpan<value_type, Extents, LayoutPolicy, AP> &s)
    requires(!IsStatic && !std::same_as<element_type, value_type>)
      : MDSpan(s.accessor().as_std_span(), s.extents()) {}
  template <typename AP>
  MDSpan(const MDSpan<value_type, Extents, LayoutPolicy, AP> &s)
    requires(IsStatic && !std::same_as<element_type, value_type>)
      : MDSpan(s.accessor().as_std_span()) {}

  MDSpan(element_type *s)
    requires(IsStatic)
      : ParentType(), m_accessor(std_span_type(s, std_span_type::extent)) {}

  MDSpan(const std_span_type &s)
    requires(!IsStatic && extents_type::rank() == 1)
      : ParentType(extents_type(s.size())), m_accessor(s) {}

  auto begin() noexcept { return m_accessor.begin(); }
  auto end() noexcept { return m_accessor.end(); }
  auto begin() const noexcept { return m_accessor.begin(); }
  auto end() const noexcept { return m_accessor.end(); }

private:
  accessor_type m_accessor;
};

} // namespace zipper::expression::nullary
namespace zipper::expression {

template <typename ElementType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ExpressionTraits<
    nullary::MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>>
    : public detail::DefaultExpressionTraits<ElementType, Extents>
/*: public detail::ExpressionTraits <
  views::StorageExpressionBase<zipper::storage::MDSpan<
      ElementType, Extents, LayoutPolicy, AccessorPolicy>> */
{
  using element_type = ElementType;
  using value_type = std::remove_cv_t<ElementType>;
  constexpr static bool is_const = std::is_const_v<ElementType>;
  using extents_type = Extents;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  using value_accessor_type =
      storage::SpanData<ElementType, extents_traits::static_size>;
  using layout_policy = LayoutPolicy;
  using accessor_policy = AccessorPolicy;
  using mapping_type = typename layout_policy::template mapping<extents_type>;
  constexpr static bool is_writable = !is_const;
  constexpr static bool is_coefficient_consistent = true;
};
} // namespace zipper::expression
#endif
