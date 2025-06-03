#if !defined(ZIPPER_MATRIX_SPAN_HPP)
#define ZIPPER_MATRIX_SPAN_HPP

#include "MatrixBase.hpp"
#include "storage/SpanStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, index_type Rows, index_type Cols,
          bool RowMajor = false>
class MatrixSpan
    : public MatrixBase<storage::SpanStorage<
          ValueType, zipper::extents<Rows, Cols>,
          std::conditional_t<RowMajor, std::experimental::layout_left,
                             std::experimental::layout_right>>> {
   public:
    using layout_type =
        std::conditional_t<RowMajor, std::experimental::layout_left,
                           std::experimental::layout_right>;
    using Storage = storage::SpanStorage<ValueType, zipper::extents<Rows, Cols>,
                                         layout_type>;
    using Base = MatrixBase<Storage>;

    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;
    using std_span_type = Storage::std_span_type;
    using Base::extent;
    using Base::extents;

    MatrixSpan(const std_span_type& e)
        requires(extents_traits::is_static)
        : Base(Storage(e)) {}

    MatrixSpan(const std_span_type& e)
        requires(extents_traits::rank_dynamic == 1)
        : Base(Storage(
              e, extents_type(e.size() / extents_traits::static_sub_size))) {
        assert(e.size() % extents_traits::static_sub_size == 0);
    }

    MatrixSpan(const std_span_type& e, index_type rows, index_type cols)
        requires(extents_traits::rank_dynamic == 2)
        : Base(Storage(e, extents_type(rows, cols))) {}

    using Base::operator=;
};

}  // namespace zipper

namespace zipper::views {

template <typename ValueType, index_type Rows, index_type Cols, bool RowMajor>
struct detail::ViewTraits<MatrixSpan<ValueType, Rows, Cols, RowMajor>>
    : public detail::ViewTraits<zipper::storage::SpanStorage<
          ValueType, zipper::extents<Rows, Cols>,
          std::conditional_t<RowMajor, std::experimental::layout_left,
                             std::experimental::layout_right>>> {};
}  // namespace zipper::views
#endif
