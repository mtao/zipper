#if !defined(ZIPPER_TENSOR_SPAN_HPP)
#define ZIPPER_TENSOR_SPAN_HPP

#include "TensorBase.hpp"
#include "storage/SpanStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, concepts::ExtentsType Extents, bool LeftMajor>
class TensorSpan_
    : public TensorBase<storage::SpanStorage<
          ValueType, Extents,
          std::conditional_t<LeftMajor, std::experimental::layout_left,
                             std::experimental::layout_right>>> {
   public:
    using layout_type =
        std::conditional_t<LeftMajor, std::experimental::layout_left,
                           std::experimental::layout_right>;
    using Storage = storage::SpanStorage<ValueType, Extents, layout_type>;
    using Base = TensorBase<Storage>;

    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;
    using std_span_type = Storage::std_span_type;
    using Base::extent;
    using Base::extents;

    TensorSpan_(const std_span_type& e)
        requires(extents_traits::is_static)
        : Base(Storage(e)) {}

    TensorSpan_(const std_span_type& e)
        requires(!extents_traits::is_static && extents_type::rank() == 1)
        : Base(Storage(e, zipper::extents(e.size()))) {}

    template <typename... Args>
    TensorSpan_(const std_span_type& e, Args&&... args)
        requires((std::is_convertible_v<Args, index_type> && ...) && sizeof...(args) == extents_type::rank())
        : Base(Storage(e, Extents(std::forward<Args>(args)...))) {}

    using Base::operator=;
};

template <typename ValueType, index_type... Indxs>
using TensorSpan = TensorSpan_<ValueType, zipper::extents<Indxs...>>;
}  // namespace zipper

#endif
