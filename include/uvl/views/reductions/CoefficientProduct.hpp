#if !defined(UVL_VIEWS_COEFFICIENTPRODUCT_HPP)
#define UVL_VIEWS_COEFFICIENTPRODUCT_HPP

#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/detail/all_extents_indices.hpp"
#include "uvl/views/detail/ViewTraits.hpp"

namespace uvl::views {
namespace reductions {

template <concepts::ViewDerived View>
class CoefficientProduct {
   public:
    using self_type = CoefficientProduct<View>;
    using view_type = View;
    using view_traits = uvl::views::detail::ViewTraits<view_type>;
    using value_type = typename View::value_type;

    CoefficientProduct(View&& v) : m_view(v) {}
    CoefficientProduct(const View& v) : m_view(v) {}

    CoefficientProduct(CoefficientProduct&& v) = default;
    CoefficientProduct(const CoefficientProduct& v) = default;

    value_type operator()() const {
        value_type v = 1.0;
        for (const auto& i :
             uvl::detail::all_extents_indices(m_view.extents())) {
            v *= m_view(i);
        }
        return v;
    }

   private:
    const View& m_view;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

template <concepts::ViewDerived View>
CoefficientProduct(const View&) -> CoefficientProduct<View>;

}  // namespace reductions
}  // namespace uvl::views
#endif
