#if !defined(UVL_VIEWS_COEFFICIENTSUM_HPP)
#define UVL_VIEWS_COEFFICIENTSUM_HPP

#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/detail/extents/all_extents_indices.hpp"
#include "uvl/views/detail/ViewTraits.hpp"

namespace uvl::views {
namespace reductions {

template <concepts::ViewDerived View>
class CoefficientSum {
   public:
    using self_type = CoefficientSum<View>;
    using view_type = View;
    using view_traits = uvl::views::detail::ViewTraits<view_type>;
    using value_type = typename View::value_type;

    CoefficientSum(View&& v) : m_view(v) {}
    CoefficientSum(const View& v) : m_view(v) {}
    CoefficientSum& operator=(View&& v) { m_view = v; }
    CoefficientSum& operator=(const View& v) { m_view = v; }

    CoefficientSum(CoefficientSum&& v) = default;
    CoefficientSum(const CoefficientSum& v) = default;
    CoefficientSum& operator=(CoefficientSum&& v) = default;
    CoefficientSum& operator=(const CoefficientSum& v) = default;

    value_type operator()() const {
        value_type v = 0.0;
        for (const auto& i :
             uvl::detail::extents::all_extents_indices(m_view.extents())) {
            v += m_view(i);
        }
        return v;
    }

   private:
    const View& m_view;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

template <concepts::ViewDerived View>
CoefficientSum(const View&) -> CoefficientSum<View>;

}  // namespace reductions
}  // namespace uvl::views
#endif
