

#if !defined(UVL_STORAGE_PLAINOBJECTSTORAGE_HPP)
#define UVL_STORAGE_PLAINOBJECTSTORAGE_HPP

#include "PlainObjectAccessor.hpp"
#include "uvl/detail//ExtentsTraits.hpp"
#include "uvl/views/PlainObjectViewBase.hpp"

namespace uvl::storage {

template <typename ValueType, typename Extents,
          typename LayoutPolicy = uvl::default_layout_policy,
          typename AccessorPolicy = uvl::default_accessor_policy<ValueType>>
class PlainObjectStorage
    : public views::PlainObjectViewBase<PlainObjectStorage<
          ValueType, Extents, LayoutPolicy, AccessorPolicy>> {
   public:
    using ParentType = views::PlainObjectViewBase<
        PlainObjectStorage<ValueType, Extents, LayoutPolicy, AccessorPolicy>>;
    using value_type = ValueType;
    using extents_type = Extents;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
    constexpr static bool IsStatic =
        uvl::detail::ExtentsTraits<extents_type>::is_static;

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

    template <typename... Args>
    PlainObjectStorage(const extents_type& extents, Args&&... args)
        : ParentType(extents), m_accessor(std::forward<Args>(args)...) {}

    template <concepts::ExtentsType E2>
    void resize(const E2& e)
        requires(extents_traits::template is_convertable_from<
                 E2>() && 
            !IsStatic)
    {
        static_assert(E2::rank() != 0);
        this->resize_extents(e);
        m_accessor.container().resize(extents_traits::size(e));
    }

   private:
    accessor_type m_accessor;
};

}  // namespace uvl::storage
namespace uvl::views {

template <typename ValueType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ViewTraits<uvl::storage::PlainObjectStorage<
    ValueType, Extents, LayoutPolicy, AccessorPolicy>>
    : public detail::DefaultViewTraits<ValueType, Extents>
/*: public detail::ViewTraits <
  views::PlainObjectViewBase<uvl::storage::PlainObjectStorage<
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
}  // namespace uvl::views
#endif
