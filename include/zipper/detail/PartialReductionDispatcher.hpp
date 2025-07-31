#if !defined(ZIPPER_DETAIL_PARTIALREDUCTIONDISPATCHER_HPP)
#define ZIPPER_DETAIL_PARTIALREDUCTIONDISPATCHER_HPP
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/views/unary/detail/PartialReductionViewDispatcher.hpp"

namespace zipper::detail {


template <template <concepts::QualifiedViewDerived> typename DerivedT, zipper::concepts::QualifiedViewDerived ViewType, rank_type... Indices>
class PartialReductionDispatcher {
   public:
    PartialReductionDispatcher(ViewType& v) : m_view_dispatcher(v) {}
    auto sum() const {

        DerivedT a = m_view_dispatcher.sum();
        return a;
    }
    template <index_type P = 2>
    auto norm() const {
        DerivedT a = m_view_dispatcher.norm();
        return a;
    }
    template <index_type P = 2>
    auto norm_powered() const {
        DerivedT a = m_view_dispatcher.norm_powered();
        return a;
    }

   private:
    views::unary::detail::PartialReductionViewDispatcher<ViewType,Indices...> m_view_dispatcher;
};

}
#endif
