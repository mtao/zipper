#if !defined(ZIPPER_EXPRESSION_NULLARY_MDSPAN_HPP)
#define ZIPPER_EXPRESSION_NULLARY_MDSPAN_HPP

#include "LinearLayoutExpression.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/detail/NoAliasTraits.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"
#include "zipper/storage/AccessorData.hpp"
#include "zipper/storage/SpanData.hpp"
#include "zipper/storage/layout_types.hpp"
#include <array>
#include <span>
#include <vector>

namespace zipper::expression::nullary {

namespace _detail_mdspan {
    /// Default accessors read a raw buffer (fast linear-array path); any
    /// other accessor is honored by reading through it.
    template <typename ElementType, typename Extents, typename AccessorPolicy>
    using storage_t = std::conditional_t<
        std::is_same_v<AccessorPolicy, default_accessor_policy<ElementType>>,
        storage::SpanData<ElementType,
                          zipper::detail::ExtentsTraits<Extents>::static_size>,
        storage::AccessorData<
            ElementType,
            AccessorPolicy,
            zipper::detail::ExtentsTraits<Extents>::static_size>>;
} // namespace _detail_mdspan

/// Non-owning view of external memory, mirroring
/// std::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>.
/// Elements are read (and, for reference-returning accessors, written)
/// through AccessorPolicy.
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<ElementType>>
class MDSpan
  : public LinearLayoutExpression<
        _detail_mdspan::storage_t<ElementType, Extents, AccessorPolicy>,
        Extents,
        LayoutPolicy,
        MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>> {
    using base_type = LinearLayoutExpression<
        _detail_mdspan::storage_t<ElementType, Extents, AccessorPolicy>,
        Extents,
        LayoutPolicy,
        MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>>;
    using base_type::base_type;

    constexpr static bool is_default_accessor =
        std::is_same_v<AccessorPolicy, default_accessor_policy<ElementType>>;

    using extents_traits = zipper::detail::ExtentsTraits<Extents>;

  public:
    using self_type =
        MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
    using extents_type = Extents;

    /// Converting constructor: allows MDSpan<T,...> -> MDSpan<const T,...>
    template <typename OtherElementType,
              typename OtherLayoutPolicy,
              typename OtherAccessorPolicy>
    MDSpan(const MDSpan<OtherElementType,
                        Extents,
                        OtherLayoutPolicy,
                        OtherAccessorPolicy> &other)
        requires(std::is_const_v<ElementType>
                 && !std::is_const_v<OtherElementType>
                 && std::is_same_v<std::remove_cv_t<ElementType>,
                                   std::remove_cv_t<OtherElementType>>
                 && std::is_constructible_v<
                     typename base_type::linear_accessor_type,
                     const typename MDSpan<OtherElementType,
                                           Extents,
                                           OtherLayoutPolicy,
                                           OtherAccessorPolicy>::
                         linear_accessor_type &>)
      : base_type(
            typename base_type::linear_accessor_type(other.linear_accessor()),
            other.extents()) {}

    /// Constructor from std::span with matching static extent
    template <std::size_t N>
    MDSpan(std::span<ElementType, N> s)
        requires(is_default_accessor && extents_traits::is_static && N == extents_traits::static_size)
      : base_type(typename base_type::linear_accessor_type(s)) {}

    /// Constructor from std::span with dynamic extent (rank-1 only)
    MDSpan(std::span<ElementType, std::dynamic_extent> s)
        requires(is_default_accessor && !extents_traits::is_static && extents_type::rank() == 1)
      : base_type(typename base_type::linear_accessor_type(s),
                  extents_type(s.size())) {}

    /// Explicit constructor from std::span with dynamic extent into static
    /// extents. The size is checked at runtime (ZIPPER_ASSERT).
    explicit MDSpan(std::span<ElementType, std::dynamic_extent> s)
        requires(is_default_accessor && extents_traits::is_static)
      : base_type(typename base_type::linear_accessor_type(
            std::span<ElementType, extents_traits::static_size>(
                s.data(),
                extents_traits::static_size))) {
        ZIPPER_ASSERT(s.size() == extents_traits::static_size);
    }

    /// Constructor from std::array (mutable reference, static extents)
    template <std::size_t N>
    MDSpan(std::array<std::remove_const_t<ElementType>, N> &arr)
        requires(is_default_accessor && !std::is_const_v<ElementType> && extents_traits::is_static
                 && N == extents_traits::static_size)
      : MDSpan(std::span<ElementType, N>(arr)) {}

    /// Constructor from const std::array (static extents)
    template <std::size_t N>
    MDSpan(const std::array<std::remove_const_t<ElementType>, N> &arr)
        requires(is_default_accessor && std::is_const_v<ElementType> && extents_traits::is_static
                 && N == extents_traits::static_size)
      : MDSpan(std::span<ElementType, N>(arr)) {}

    /// Constructor from std::vector (mutable reference, dynamic extents,
    /// rank-1)
    template <typename Alloc>
    MDSpan(std::vector<std::remove_const_t<ElementType>, Alloc> &vec)
        requires(is_default_accessor && !std::is_const_v<ElementType> && !extents_traits::is_static
                 && extents_type::rank() == 1)
      : MDSpan(std::span<ElementType, std::dynamic_extent>(vec)) {}

    /// Constructor from const std::vector (dynamic extents, rank-1)
    template <typename Alloc>
    MDSpan(const std::vector<std::remove_const_t<ElementType>, Alloc> &vec)
        requires(is_default_accessor && std::is_const_v<ElementType> && !extents_traits::is_static
                 && extents_type::rank() == 1)
      : MDSpan(std::span<ElementType, std::dynamic_extent>(vec)) {}

    /// Constructor from std::mdspan (or zipper::mdspan) with the same
    /// layout; the data handle, mapping and accessor are all taken from it.
    template <typename OtherElementType, typename OtherAccessor>
        requires(std::is_constructible_v<AccessorPolicy, const OtherAccessor &>
                 && std::is_convertible_v<
                     typename OtherAccessor::data_handle_type,
                     typename AccessorPolicy::data_handle_type>)
    MDSpan(zipper::mdspan<OtherElementType, Extents, LayoutPolicy, OtherAccessor>
               m)
      : base_type(make_linear_accessor(m), m.extents()) {}

    /// Constructor mirroring std::mdspan's (pointer, extents...) form:
    /// builds the mdspan in place, so callers don't have to spell out the
    /// mdspan type.  E.g. for extents (3, dynamic):
    ///     MatrixX3::const_span_type m(ptr, n);
    template <typename... IndexTypes>
        requires(sizeof...(IndexTypes) > 0
                 && std::is_constructible_v<
                     zipper::mdspan<ElementType,
                                    Extents,
                                    LayoutPolicy,
                                    AccessorPolicy>,
                     typename AccessorPolicy::data_handle_type,
                     IndexTypes...>)
    explicit MDSpan(typename AccessorPolicy::data_handle_type data,
                    IndexTypes... exts)
      : MDSpan(zipper::mdspan<ElementType,
                              Extents,
                              LayoutPolicy,
                              AccessorPolicy>(std::move(data), exts...)) {}

    /// Fully static extents: the pointer alone determines the view.
    explicit MDSpan(typename AccessorPolicy::data_handle_type data)
        requires(extents_traits::is_static)
      : MDSpan(zipper::mdspan<ElementType,
                              Extents,
                              LayoutPolicy,
                              AccessorPolicy>(std::move(data))) {}

    template <zipper::concepts::Expression V>
    void assign(const V &v)
        requires(!std::is_const_v<ElementType>
                 && zipper::utils::extents::assignable_extents_v<
                     typename V::extents_type,
                     extents_type>)
    {
        expression::detail::AssignHelper<V, self_type>::assign(v, *this);
    }

    /// MDSpan borrows external memory — make_owned() deep-copies into an
    /// MDArray. Defined out-of-line (after MDArray.hpp is included) because
    /// MDArray depends on MDSpan and not vice versa.
    auto make_owned() const;

  private:
    template <typename OtherElementType, typename OtherAccessor>
    static auto make_linear_accessor(
        const zipper::mdspan<OtherElementType,
                             Extents,
                             LayoutPolicy,
                             OtherAccessor> &m) ->
        typename base_type::linear_accessor_type {
        using linear_accessor_type = typename base_type::linear_accessor_type;
        const auto size = m.mapping().required_span_size();
        if constexpr (is_default_accessor) {
            return linear_accessor_type(
                std::span<ElementType, extents_traits::static_size>(
                    m.data_handle(), size));
        } else {
            return linear_accessor_type(
                m.data_handle(), size, AccessorPolicy(m.accessor()));
        }
    }
};
} // namespace zipper::expression::nullary

namespace zipper::expression {
/// ExpressionTraits for MDSpan forwards to the LinearLayoutExpression traits.
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ExpressionTraits<
    nullary::MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>>
  : public detail::ExpressionTraits<nullary::LinearLayoutExpression<
        nullary::_detail_mdspan::
            storage_t<ElementType, Extents, AccessorPolicy>,
        Extents,
        LayoutPolicy,
        nullary::MDSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>>> {
    /// MDSpan borrows external memory via a span, so it stores references.
    constexpr static bool stores_references = true;
};
} // namespace zipper::expression

namespace zipper::detail {
// Mutable views over external storage support noalias destinations when
// their elements are real references.
template <typename T, typename E, typename L, typename A>
inline constexpr bool
    supports_noalias_v<expression::nullary::MDSpan<T, E, L, A>> =
        !std::is_const_v<T>
        && std::is_same_v<typename A::reference, T &>;
} // namespace zipper::detail

#endif
