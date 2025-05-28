#if !defined(ZIPPER_STORAGE_DENSEACCESSOR_HPP)
#define ZIPPER_STORAGE_DENSEACCESSOR_HPP

#include "SpanData.hpp"
#include "concepts/DataLike.hpp"
#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/views/nullary/DenseStorageViewBase.hpp"

namespace zipper::storage {

// SpanStorage predeclares the defaults now?
// template <typename ValueType, typename Extents,
//           typename LayoutPolicy = zipper::default_layout_policy,
//           typename AccessorPolicy =
//           zipper::default_accessor_policy<ValueType>>
template <concepts::DataLike DataType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
class DenseAccessor
    : public views::nullary::DenseStorageViewBase<
          DenseAccessor<DataType, Extents, LayoutPolicy, AccessorPolicy>> {
   public:
    using ParentType = views::nullary::DenseStorageViewBase<
        DenseAccessor<DataType, Extents, LayoutPolicy, AccessorPolicy>>;
    using data_type = DataType;
    using value_type = DataType::value_type;
    using extents_type = Extents;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    constexpr static bool IsStatic =
        zipper::detail::ExtentsTraits<extents_type>::is_static;

    using span_type = DenseAccessor<SpanData<value_type, DataType::static_size>,
                                    Extents, LayoutPolicy, AccessorPolicy>;

    const data_type& linear_access() const { return m_data; }
    data_type& linear_access() { return m_data; }
    using ParentType::assign;
    using ParentType::extents;
    using ParentType::size;

    DenseAccessor()
        requires(IsStatic)
        : ParentType() {}

    DenseAccessor(const DenseAccessor&) = default;
    DenseAccessor(DenseAccessor&&) = default;
    DenseAccessor& operator=(const DenseAccessor&) = default;
    DenseAccessor& operator=(DenseAccessor&&) = default;

    DenseAccessor(const extents_type& extents)
        requires(!IsStatic)
        : ParentType(extents), m_data(extents_traits::size(extents)) {
        assert(m_data.size() == extents_traits::size(extents));
    }
    DenseAccessor(data_type&& data)
        requires(IsStatic)
        : ParentType(), m_data(std::move(data)) {
        static_assert(extents_traits::static_size == data_type::static_size);
    }

    DenseAccessor(data_type data, const extents_type& extents)
        requires(!IsStatic)
        : ParentType(extents), m_data(std::move(data)) {
        assert(m_data.size() == extents_traits::size(extents));
    }
    DenseAccessor(data_type data, const extents_type& extents)
        requires(IsStatic)
        : ParentType(extents), m_data(std::move(data)) {
        static_assert(extents_traits::static_size == data_type::static_size);
    }

    span_type as_span() {
        if constexpr (IsStatic) {
            return span_type(linear_access().as_std_span());
        } else {
            return span_type(linear_access().as_std_span(), extents());
        }
    }
    const span_type as_span() const {
        if constexpr (IsStatic) {
            return span_type(linear_access().as_std_span());
        } else {
            return span_type(linear_access().as_std_span(), extents());
        }
    }

    template <zipper::concepts::ExtentsType E2>
    void resize(const E2& e)
        requires(extents_traits::template is_convertable_from<E2>() &&
                 !IsStatic)
    {
        static_assert(E2::rank() != 0);
        this->resize_extents(e);
        m_data.container().resize(extents_traits::size(e));
    }
    using iterator_type = data_type::iterator_type;
    using const_iterator_type = data_type::const_iterator_type;

    auto begin() { return m_data.begin(); }
    auto end() { return m_data.end(); }
    auto begin() const { return m_data.begin(); }
    auto end() const { return m_data.end(); }

   private:
    data_type m_data;
};
}  // namespace zipper::storage

namespace zipper::views {

template <typename DataType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ViewTraits<zipper::storage::DenseAccessor<
    DataType, Extents, LayoutPolicy, AccessorPolicy>>
    : public detail::DefaultViewTraits<typename DataType::value_type, Extents>
/*: public detail::ViewTraits <
  views::StorageViewBase<zipper::storage::DenseStorage<
      ValueType, Extents, LayoutPolicy, AccessorPolicy>> */
{
    using value_type = typename DataType::value_type;
    using extents_type = Extents;
    using value_accessor_type = DataType;
    using layout_policy = LayoutPolicy;
    using accessor_policy = AccessorPolicy;
    using mapping_type = typename layout_policy::template mapping<extents_type>;
    constexpr static bool is_writable = true;
    constexpr static bool is_coefficient_consistent = true;
};
}  // namespace zipper::views
#endif
