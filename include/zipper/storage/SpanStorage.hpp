#if !defined(ZIPPER_STORAGE_SPANSTORAGE_HPP)
#define ZIPPER_STORAGE_SPANSTORAGE_HPP

#include "SpanData.hpp"
#include "layout_types.hpp"
#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/views/nullary/DenseStorageViewBase.hpp"

namespace zipper::storage {
template <typename ValueType, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<ValueType>>
class PlainObjectStorage;

template <typename ValueType, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<ValueType>>
class SpanStorage
    : public views::nullary::DenseStorageViewBase<
          SpanStorage<ValueType, Extents, LayoutPolicy, AccessorPolicy>> {
   public:
    using self_type = SpanStorage<ValueType, Extents, LayoutPolicy, AccessorPolicy>;
    using ParentType = views::nullary::DenseStorageViewBase<self_type>;
    using traits = views::detail::ViewTraits<self_type>;
    using value_type = typename traits::value_type;
    using extents_type = Extents;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    using std_span_type = std::span<ValueType, extents_traits::static_size>;
    constexpr static bool IsStatic =
        zipper::detail::ExtentsTraits<extents_type>::is_static;

    using accessor_type = SpanData<ValueType, extents_traits::static_size>;
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

    SpanStorage(const std_span_type& s)
        requires(!IsStatic && extents_type::rank() == 1)
        : ParentType(extents_type(s.size())), m_accessor(s) {}

    template <concepts::ExtentsType E2>
    void resize(const E2& e)
        requires(extents_traits::template is_convertable_from<E2>() &&
                 !IsStatic)
    {
        static_assert(E2::rank() != 0);
        this->resize_extents(e);
        m_accessor.container().resize(extents_traits::size(e));
    }

   private:
    accessor_type m_accessor;
};

}  // namespace zipper::storage
namespace zipper::views {

template <typename ValueType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ViewTraits<zipper::storage::SpanStorage<
    ValueType, Extents, LayoutPolicy, AccessorPolicy>>
    : public detail::DefaultViewTraits<ValueType, Extents>
/*: public detail::ViewTraits <
  views::StorageViewBase<zipper::storage::SpanStorage<
      ValueType, Extents, LayoutPolicy, AccessorPolicy>> */
{
    using value_type = std::remove_const_t<ValueType>;
    constexpr static bool is_const = std::is_const_v<ValueType>;
    using extents_type = Extents;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    using value_accessor_type =
        storage::SpanData<ValueType, extents_traits::static_size>;
    using layout_policy = LayoutPolicy;
    using accessor_policy = AccessorPolicy;
    using mapping_type = typename layout_policy::template mapping<extents_type>;
    constexpr static bool is_writable = !is_const;
    constexpr static bool is_coefficient_consistent = true;
};
}  // namespace zipper::views
#endif
