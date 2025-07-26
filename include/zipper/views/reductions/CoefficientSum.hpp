#if !defined(ZIPPER_VIEWS_COEFFICIENTSUM_HPP)
#define ZIPPER_VIEWS_COEFFICIENTSUM_HPP

#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"
#include "zipper/views/detail/ViewTraits.hpp"

namespace zipper::views {
namespace reductions {

template <zipper::concepts::ViewDerived View>
class CoefficientSum {
   public:
    using self_type = CoefficientSum<View>;
    using view_type = View;
    using view_traits = zipper::views::detail::ViewTraits<view_type>;
    using value_type = typename View::value_type;

    CoefficientSum(View&& v) : m_view(v) {}
    CoefficientSum(const View& v) : m_view(v) {}

    CoefficientSum(CoefficientSum&& v) = default;
    CoefficientSum(const CoefficientSum& v) = default;
    CoefficientSum& operator=(CoefficientSum&& v) = delete;
    CoefficientSum& operator=(const CoefficientSum& v) = delete;

    value_type operator()() const {
        value_type v = 0.0;
        for (const auto& i :
             zipper::utils::extents::all_extents_indices(m_view.extents())) {
            v += m_view(i);
        }
        return v;
    }

   private:
    const View& m_view;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

template <zipper::concepts::ViewDerived View>
CoefficientSum(const View&) -> CoefficientSum<View>;

}  // namespace reductions
}  // namespace zipper::views
#endif
