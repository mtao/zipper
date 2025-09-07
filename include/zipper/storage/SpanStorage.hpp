#if !defined(ZIPPER_STORAGE_SPANSTORAGE_HPP)
#define ZIPPER_STORAGE_SPANSTORAGE_HPP

#include "SpanData.hpp"
#include "layout_types.hpp"
#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/views/nullary/DenseStorageViewBase.hpp"

namespace zipper::storage {
template <typename ElementType, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<ElementType>>
class PlainObjectStorage;

template <typename ElementType, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<ElementType>>
class SpanStorage
    : public views::nullary::DenseStorageViewBase<
          SpanStorage<ElementType, Extents, LayoutPolicy, AccessorPolicy>> {
   public:
    using self_type =
        SpanStorage<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
    using ParentType = views::nullary::DenseStorageViewBase<self_type>;
    using traits = views::detail::ViewTraits<self_type>;
    using value_type = typename traits::value_type;
    using element_type = typename traits::element_type;
    using extents_type = Extents;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    using std_span_type = std::span<ElementType, extents_traits::static_size>;
    constexpr static bool IsStatic =
        zipper::detail::ExtentsTraits<extents_type>::is_static;

    using accessor_type = SpanData<ElementType, extents_traits::static_size>;
    const accessor_type& accessor() const { return m_accessor; }
    accessor_type& accessor() { return m_accessor; }
    const accessor_type& linear_access() const { return m_accessor; }
    accessor_type& linear_access() { return m_accessor; }
    using ParentType::assign;

    SpanStorage()
        requires(IsStatic)
        : ParentType(), m_accessor() {}

    SpanStorage(const SpanStorage&) = default;
    SpanStorage(SpanStorage&&) = default;
    SpanStorage& operator=(const SpanStorage&) = delete;
    SpanStorage& operator=(SpanStorage&&) = delete;

    SpanStorage(const std_span_type& s, const extents_type& extents)
        requires(!IsStatic)
        : ParentType(extents), m_accessor(s) {}

    SpanStorage(const std_span_type& s)
        requires(IsStatic)
        : ParentType(), m_accessor(s) {}

    SpanStorage(const std::span<value_type, extents_traits::static_size>& s,
                const extents_type& extents)
        requires(!IsStatic && !std::same_as<element_type, value_type>)
        : ParentType(extents), m_accessor(s) {}

    SpanStorage(const std::span<value_type, extents_traits::static_size>& s)
        requires(IsStatic && !std::same_as<element_type, value_type>)
        : ParentType(), m_accessor(s) {}

    template <typename AP>
    SpanStorage(const SpanStorage<value_type, Extents, LayoutPolicy, AP>& s)
        requires(!IsStatic && !std::same_as<element_type, value_type>)
        : SpanStorage(s.accessor().as_std_span(), s.extents()) {}
    template <typename AP>
    SpanStorage(const SpanStorage<value_type, Extents, LayoutPolicy, AP>& s)
        requires(IsStatic && !std::same_as<element_type, value_type>)
        : SpanStorage(s.accessor().as_std_span()) {}

    // SpanStorage(const std::span<value_type, std_span_type& s, const
    // extents_type& extents)
    //     requires(!IsStatic)
    //     : ParentType(extents), m_accessor(s) {}

    // SpanStorage(const std_span_type& s)
    //     requires(IsStatic)
    //     : ParentType(), m_accessor(s) {}

    SpanStorage(element_type* s)
        requires(IsStatic)
        : ParentType(), m_accessor(std_span_type(s, std_span_type::extent)) {}

    SpanStorage(const std_span_type& s)
        requires(!IsStatic && extents_type::rank() == 1)
        : ParentType(extents_type(s.size())), m_accessor(s) {}

    auto begin() noexcept { return m_accessor.begin(); }
    auto end() noexcept { return m_accessor.end(); }
    auto begin() const noexcept { return m_accessor.begin(); }
    auto end() const noexcept { return m_accessor.end(); }

   private:
    accessor_type m_accessor;
};

}  // namespace zipper::storage
namespace zipper::views {

template <typename ElementType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ViewTraits<zipper::storage::SpanStorage<
    ElementType, Extents, LayoutPolicy, AccessorPolicy>>
    : public detail::DefaultViewTraits<ElementType, Extents>
/*: public detail::ViewTraits <
  views::StorageViewBase<zipper::storage::SpanStorage<
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
}  // namespace zipper::views
#endif
