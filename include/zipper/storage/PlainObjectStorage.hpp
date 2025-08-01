#if !defined(ZIPPER_STORAGE_PLAINOBJECTSTORAGE_HPP)
#define ZIPPER_STORAGE_PLAINOBJECTSTORAGE_HPP

#include "PlainObjectAccessor.hpp"
#include "SpanStorage.hpp"
#include "layout_types.hpp"
#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/views/nullary/DenseStorageViewBase.hpp"

namespace zipper::storage {
// defaults are stored in SpanStorage
template <typename ValueType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
class PlainObjectStorage
    : public views::nullary::DenseStorageViewBase<PlainObjectStorage<
          ValueType, Extents, LayoutPolicy, AccessorPolicy>> {
   public:
    using ParentType = views::nullary::DenseStorageViewBase<
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
    const accessor_type& linear_access() const { return m_accessor; }
    accessor_type& linear_access() { return m_accessor; }
    using ParentType::assign;

    PlainObjectStorage() : ParentType(), m_accessor() {}

    PlainObjectStorage(const PlainObjectStorage&) = default;
    PlainObjectStorage(PlainObjectStorage&&) = default;
    PlainObjectStorage& operator=(const PlainObjectStorage&) = default;
    PlainObjectStorage& operator=(PlainObjectStorage&&) = default;

    PlainObjectStorage(const extents_type& extents)
        requires(!IsStatic)
        : ParentType(extents), m_accessor(extents_traits::size(extents)) {}
    PlainObjectStorage(const extents_type& extents)
        requires(IsStatic)
        : ParentType(extents), m_accessor() {}

    span_type as_span() {
        if constexpr (IsStatic) {
            return span_type(accessor().as_stl_span());
        } else {
            const extents_type& e = ParentType::extents();
            return span_type(accessor().as_stl_span(), e);
        }
    }
    const span_type as_span() const {
        if constexpr (IsStatic) {
            return span_type(accessor().as_stl_span());
        } else {
            const extents_type& e = ParentType::extents();
            return span_type(accessor().as_stl_span(), e);
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
    using iterator_type = accessor_type::iterator_type;
    using const_iterator_type = accessor_type::const_iterator_type;
    auto begin() { return m_accessor.begin(); }
    auto end() { return m_accessor.end(); }
    auto begin() const { return m_accessor.begin(); }
    auto end() const { return m_accessor.end(); }

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
    constexpr static bool is_resizable = extents_type::rank_dynamic() > 0;
};
}  // namespace zipper::views
#endif
