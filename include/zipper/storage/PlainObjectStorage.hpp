#if !defined(ZIPPER_STORAGE_PLAINOBJECTSTORAGE_HPP)
#define ZIPPER_STORAGE_PLAINOBJECTSTORAGE_HPP

#include "PlainObjectAccessor.hpp"
#include "SpanStorage.hpp"
#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/views/StorageViewBase.hpp"

namespace zipper::storage {

// SpanStorage predeclares the defaults now?
// template <typename ValueType, typename Extents,
//           typename LayoutPolicy = zipper::default_layout_policy,
//           typename AccessorPolicy =
//           zipper::default_accessor_policy<ValueType>>
template <typename ValueType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
class PlainObjectStorage
    : public views::StorageViewBase<PlainObjectStorage<
          ValueType, Extents, LayoutPolicy, AccessorPolicy>> {
   public:
    using ParentType = views::StorageViewBase<
        PlainObjectStorage<ValueType, Extents, LayoutPolicy, AccessorPolicy>>;
    using value_type = ValueType;
    using extents_type = Extents;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    constexpr static bool IsStatic =
        zipper::detail::ExtentsTraits<extents_type>::is_static;

    using span_type =
        SpanStorage<ValueType, Extents, LayoutPolicy, AccessorPolicy>;

    using accessor_type = PlainObjectAccessor<value_type, extents_type>;
    const accessor_type& accessor() const { return m_accessor; }
    accessor_type& accessor() { return m_accessor; }
    using ParentType::assign;

    /*
    template <typename... Args>
    PlainObjectStorage(Args&&... args)
        requires(IsStatic)
        : m_accessor(std::forward<Args>(args)...) {}
    */
    PlainObjectStorage()
        requires(IsStatic)
        : ParentType() {}

    PlainObjectStorage(const PlainObjectStorage&) = default;
    PlainObjectStorage(PlainObjectStorage&&) = default;
    PlainObjectStorage& operator=(const PlainObjectStorage&) = default;
    PlainObjectStorage& operator=(PlainObjectStorage&&) = default;

    PlainObjectStorage(const extents_type& extents)
        requires(!IsStatic)
        : ParentType(extents), m_accessor(extents_traits::size(extents)) {}
    PlainObjectStorage(const extents_type& extents)
        requires(IsStatic)
        : ParentType(extents) {}

    span_type as_span() {
        if constexpr (IsStatic) {
            return span_type(accessor().span());
        } else {
            return span_type(accessor().span(), extents());
        }
    }
    const span_type as_span() const {
        if constexpr (IsStatic) {
            return span_type(accessor().span());
        } else {
            return span_type(accessor().span(), extents());
        }
    }

    template <typename... Args>
    PlainObjectStorage(const extents_type& extents, Args&&... args)
        : ParentType(extents), m_accessor(std::forward<Args>(args)...) {}

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
struct detail::ViewTraits<zipper::storage::PlainObjectStorage<
    ValueType, Extents, LayoutPolicy, AccessorPolicy>>
    : public detail::DefaultViewTraits<ValueType, Extents>
/*: public detail::ViewTraits <
  views::StorageViewBase<zipper::storage::PlainObjectStorage<
      ValueType, Extents, LayoutPolicy, AccessorPolicy>> */
{
    using value_type = ValueType;
    using extents_type = Extents;
    using value_accessor_type =
        storage::PlainObjectAccessor<ValueType, Extents>;
    using layout_policy = LayoutPolicy;
    using accessor_policy = AccessorPolicy;
    using mapping_type = typename layout_policy::template mapping<extents_type>;
    constexpr static bool is_writable = true;
    constexpr static bool is_coefficient_consistent = true;
};
}  // namespace zipper::views
#endif
