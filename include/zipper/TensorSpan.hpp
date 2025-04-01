#if !defined(ZIPPER_TENSOR_SPAN_HPP)
#define ZIPPER_TENSOR_SPAN_HPP

#include "TensorBase.hpp"
#include "storage/SpanStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, bool LeftMajor = false, index_type... Dims>
class TensorSpan
    : public TensorBase<storage::SpanStorage<
          ValueType, zipper::extents<Dims...>,
          std::conditional_t<LeftMajor, std::experimental::layout_left,
                             std::experimental::layout_right>>> {
   public:
    using layout_type =
        std::conditional_t<LeftMajor, std::experimental::layout_left,
                           std::experimental::layout_right>;
    using Storage = storage::SpanStorage<ValueType, zipper::extents<Dims...>,
                                         layout_type>;
    using Base = TensorBase<Storage>;

    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;
    using std_span_type = Storage::std_span_type;
    using Base::extent;
    using Base::extents;

    TensorSpan(const std_span_type& e)
        requires(extents_traits::is_static)
        : Base(Storage(e)) {}

    template <typename... Args>
    TensorSpan(const std_span_type& e, Args&&... args)
        requires((std::is_convertible_v<Args, index_type> && ...))
        : Base(Storage(e,zipper::extents<Dims...>(std::forward<Args>(args)...))) {}

    using Base::operator=;
};

}  // namespace zipper

namespace zipper::views {

template <typename ValueType, index_type Rows, index_type Cols, bool RowMajor>
struct detail::ViewTraits<TensorSpan<ValueType, Rows, Cols, RowMajor>>
    : public detail::ViewTraits<zipper::storage::SpanStorage<
          ValueType, zipper::extents<Rows, Cols>,
          std::conditional_t<RowMajor, std::experimental::layout_left,
                             std::experimental::layout_right>>> {};
}  // namespace zipper::views
#endif
