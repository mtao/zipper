#if !defined(ZIPPER_VECTOR_SPAN_HPP)
#define ZIPPER_VECTOR_SPAN_HPP

#include "VectorBase.hpp"
#include "storage/SpanStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, index_type Rows = std::dynamic_extent>
class VectorSpan : public VectorBase<
                       storage::SpanStorage<ValueType, zipper::extents<Rows>>> {
   public:
    using Storage = storage::SpanStorage<ValueType, zipper::extents<Rows>>;
    using Base = VectorBase<Storage>;

    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;
    using std_span_type = Storage::std_span_type;
    using Base::extent;
    using Base::extents;

    VectorSpan(const std_span_type& e)
        requires(extents_traits::is_static)
        : Base(Storage(e)) {}

    VectorSpan(const std_span_type& e)
        requires(extents_traits::is_dynamic)
        : Base(Storage(e, extents_type(e.size()))) {}
    using Base::operator=;
};

template <typename T, std::size_t size>
VectorSpan(const std::array<T,size>& arr) -> VectorSpan<T,size>;

template <typename T, typename Alloc>
VectorSpan(const std::vector<T, Alloc>& arr) -> VectorSpan<T, std::dynamic_extent>;

template <typename T, std::size_t size>
VectorSpan(const std::extent<T, size>& arr) -> VectorSpan<T, size>;


}  // namespace zipper

#endif
