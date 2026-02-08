#if !defined(ZIPPER_EXPRESSION_NULLARY_MDSPAN_HPP)
#define ZIPPER_EXPRESSION_NULLARY_MDSPAN_HPP

#include "LinearLayoutExpression.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"
#include "zipper/storage/SpanData.hpp"
#include "zipper/storage/layout_types.hpp"
#include <array>
#include <span>
#include <vector>

namespace zipper::expression::nullary {

template <typename ElementType, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<ElementType>>
class MDSpan
    : public LinearLayoutExpression<
          storage::SpanData<
              ElementType, zipper::detail::ExtentsTraits<Extents>::static_size>,
          Extents, LayoutPolicy, AccessorPolicy,
          MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>> {

  using base_type = LinearLayoutExpression<
      storage::SpanData<ElementType,
                        zipper::detail::ExtentsTraits<Extents>::static_size>,
      Extents, LayoutPolicy, AccessorPolicy,
      MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>>;
  using base_type::base_type;

  using extents_traits = zipper::detail::ExtentsTraits<Extents>;

public:
  using self_type = MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
  using extents_type = Extents;

  /// Converting constructor: allows MDSpan<T,...> -> MDSpan<const T,...>
  template <typename OtherElementType, typename OtherLayoutPolicy,
            typename OtherAccessorPolicy>
  MDSpan(const MDSpan<OtherElementType, Extents, OtherLayoutPolicy,
                      OtherAccessorPolicy> &other)
    requires(std::is_const_v<ElementType> &&
             !std::is_const_v<OtherElementType> &&
             std::is_same_v<std::remove_cv_t<ElementType>,
                            std::remove_cv_t<OtherElementType>>)
      : base_type(other) {}

  /// Constructor from std::span with matching static extent
  template <std::size_t N>
  MDSpan(std::span<ElementType, N> s)
    requires(extents_traits::is_static && N == extents_traits::static_size)
      : base_type(typename base_type::linear_accessor_type(s)) {}

  /// Constructor from std::span with dynamic extent (rank-1 only)
  MDSpan(std::span<ElementType, std::dynamic_extent> s)
    requires(!extents_traits::is_static && extents_type::rank() == 1)
      : base_type(typename base_type::linear_accessor_type(s),
                  extents_type(s.size())) {}

  /// Constructor from std::array (mutable reference, static extents)
  template <std::size_t N>
  MDSpan(std::array<std::remove_const_t<ElementType>, N> &arr)
    requires(!std::is_const_v<ElementType> && extents_traits::is_static &&
             N == extents_traits::static_size)
      : MDSpan(std::span<ElementType, N>(arr)) {}

  /// Constructor from const std::array (static extents)
  template <std::size_t N>
  MDSpan(const std::array<std::remove_const_t<ElementType>, N> &arr)
    requires(std::is_const_v<ElementType> && extents_traits::is_static &&
             N == extents_traits::static_size)
      : MDSpan(std::span<ElementType, N>(arr)) {}

  /// Constructor from std::vector (mutable reference, dynamic extents, rank-1)
  template <typename Alloc>
  MDSpan(std::vector<std::remove_const_t<ElementType>, Alloc> &vec)
    requires(!std::is_const_v<ElementType> && !extents_traits::is_static &&
             extents_type::rank() == 1)
      : MDSpan(std::span<ElementType, std::dynamic_extent>(vec)) {}

  /// Constructor from const std::vector (dynamic extents, rank-1)
  template <typename Alloc>
  MDSpan(const std::vector<std::remove_const_t<ElementType>, Alloc> &vec)
    requires(std::is_const_v<ElementType> && !extents_traits::is_static &&
             extents_type::rank() == 1)
      : MDSpan(std::span<ElementType, std::dynamic_extent>(vec)) {}

  template <concepts::Expression V>
  void assign(const V &v)
    requires(!std::is_const_v<ElementType> &&
             zipper::utils::extents::assignable_extents_v<
                 typename V::extents_type, extents_type>)
  {
    expression::detail::AssignHelper<V, self_type>::assign(v, *this);
  }
};
} // namespace zipper::expression::nullary

namespace zipper::expression {
/// ExpressionTraits for MDSpan forwards to the LinearLayoutExpression traits.
template <typename ElementType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ExpressionTraits<nullary::MDSpan<
    ElementType, Extents, LayoutPolicy, AccessorPolicy>>
    : public detail::ExpressionTraits<nullary::LinearLayoutExpression<
          storage::SpanData<ElementType,
                            zipper::detail::ExtentsTraits<Extents>::static_size>,
          Extents, LayoutPolicy, AccessorPolicy,
          nullary::MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>>> {
};
} // namespace zipper::expression

#endif
