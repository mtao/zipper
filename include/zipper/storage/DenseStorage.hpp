#if !defined(ZIPPER_STORAGE_DENSESTORAGE_HPP)
#define ZIPPER_STORAGE_DENSESTORAGE_HPP

#include "SpanStorage.hpp"
#include "concepts/DataLike.hpp"
#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/views/StorageViewBase.hpp"

namespace zipper::storage {

// SpanStorage predeclares the defaults now?
// template <typename ValueType, typename Extents,
//           typename LayoutPolicy = zipper::default_layout_policy,
//           typename AccessorPolicy =
//           zipper::default_accessor_policy<ValueType>>
template <concepts::DataLike DataType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
class DenseStorage
    : public views::StorageViewBase<
          DenseStorage<DataType, Extents, LayoutPolicy, AccessorPolicy>> {
   public:
    using ParentType = views::StorageViewBase<
        DenseStorage<DataType, Extents, LayoutPolicy, AccessorPolicy>>;
    using data_type = DataType;
    using value_type = DataType::ValueType;
    using extents_type = Extents;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    constexpr static bool IsStatic =
        zipper::detail::ExtentsTraits<extents_type>::is_static;

    using span_type = DenseStorage<SpanData<data_type, DataType::static_size>,
                                   Extents, LayoutPolicy, AccessorPolicy>;

    const data_type& accessor() const { return m_data; }
    data_type& accessor() { return m_data; }
    using ParentType::assign;

    DenseStorage()
        requires(IsStatic)
        : ParentType() {}

    DenseStorage(const DenseStorage&) = default;
    DenseStorage(DenseStorage&&) = default;
    DenseStorage& operator=(const DenseStorage&) = default;
    DenseStorage& operator=(DenseStorage&&) = default;

    DenseStorage(const extents_type& extents)
        requires(!IsStatic)
        : ParentType(extents), m_data(extents_traits::size(extents)) {}
    DenseStorage(const extents_type& extents)
        requires(IsStatic)
        : ParentType(extents) {}

    DenseStorage(data_type data, const extents_type& extents)
        requires(!IsStatic)
        : ParentType(extents), m_data(std::move(data)) {
        assert(m_data.size() == extents_traits::size(extents));
    }
    DenseStorage(data_type data, const extents_type& extents)
        requires(IsStatic)
        : ParentType(extents), m_data(std::move(data)) {
        static_assert(extents_traits::static_size == data::static_size);
    }

    span_type as_span() {
        if constexpr (IsStatic) {
            return span_type(accessor().as_stl_span());
        } else {
            return span_type(accessor().as_stl_span(), extents());
        }
    }
    const span_type as_span() const {
        if constexpr (IsStatic) {
            return span_type(accessor().as_stl_span());
        } else {
            return span_type(accessor().as_stl_span(), extents());
        }
    }

    template <concepts::ExtentsType E2>
    void resize(const E2& e)
        requires(extents_traits::template is_convertable_from<E2>() &&
                 !IsStatic)
    {
        static_assert(E2::rank() != 0);
        this->resize_extents(e);
        m_data.container().resize(extents_traits::size(e));
    }
    using iterator_type = accessor_type::iterator_type;
    using const_iterator_type = accessor_type::const_iterator_type;
    auto begin() { return m_data.begin(); }
    auto end() { return m_data.end(); }
    auto begin() const { return m_data.begin(); }
    auto end() const { return m_data.end(); }

   private:
    accessor_type m_data;
};

}  // namespace zipper::storage
namespace zipper::views {

template <typename ValueType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ViewTraits<zipper::storage::DenseStorage<
    ValueType, Extents, LayoutPolicy, AccessorPolicy>>
    : public detail::DefaultViewTraits<ValueType, Extents>
/*: public detail::ViewTraits <
  views::StorageViewBase<zipper::storage::DenseStorage<
      ValueType, Extents, LayoutPolicy, AccessorPolicy>> */
{
    using value_type = ValueType;
    using extents_type = Extents;
    using value_accessor_type = storage::DenseAccessor<ValueType, Extents>;
    using layout_policy = LayoutPolicy;
    using accessor_policy = AccessorPolicy;
    using mapping_type = typename layout_policy::template mapping<extents_type>;
    constexpr static bool is_writable = true;
    constexpr static bool is_coefficient_consistent = true;
};
}  // namespace zipper::views
#endif
