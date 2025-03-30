#if !defined(ZIPPER_VIEWS_UNARY_DETAIL_SLICEVIWER_HPP)
#define ZIPPER_VIEWS_UNARY_DETAIL_SLICEVIWER_HPP
#include "zipper/views/unary/SliceView.hpp"

namespace zipper::views::unary::detail {

template <template <concepts::ViewDerived> typename ZipperBaseDerived_,
          concepts::ViewDerived ViewType_, typename... Slices_>
struct SliceEnumerationTraits {
    template <concepts::ViewDerived T>
    using ZipperBaseDerived = ZipperBaseDerived_<T>;
    using ViewType = ViewType_;
    using slice_type = std::tuple<Slices_...>;
};

template <template <concepts::ViewDerived> typename ZipperBaseDerived,
          concepts::ViewDerived ViewType, bool IsConst, typename... Slices>
    requires(concepts::SlicePackLike<Slices...>)
class SliceViewer {
    using traits =
        SliceEnumerationTraits<ZipperBaseDerived, ViewType, Slices...>;
    using slice_type = typename traits::slice_type;

    SliceViewer(const SliceViewer& o)
        : m_slice(o.m_slice().view(), m_slice.m_slices) {}

    slice_type create_slice() {}

   private:
    slice_type m_slice;
};

}  // namespace zipper::views::unary::detail

#endif
