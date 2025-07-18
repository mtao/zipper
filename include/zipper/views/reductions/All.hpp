#if !defined(ZIPPER_VIEWS_ALL_HPP)
#define ZIPPER_VIEWS_ALL_HPP
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"
#include "zipper/views/detail/ViewTraits.hpp"

namespace zipper::views {
namespace reductions {

template <concepts::ViewDerived View>
class All {
   public:
    using self_type = All<View>;
    using view_type = View;
    using view_traits = zipper::views::detail::ViewTraits<view_type>;
    using value_type = bool;  // typename View::value_type;

    All(View&& v) : m_view(v) {}
    All(const View& v) : m_view(v) {}

    All(All&& v) = default;
    All(const All& v) = default;

    value_type operator()() const {
        for (const auto& i :
             zipper::utils::extents::all_extents_indices(m_view.extents())) {
            if (!value_type(m_view(i))) {
                return false;
            }
        }
        return true;
    }

   private:
    const View& m_view;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

template <concepts::ViewDerived View>
All(const View&) -> All<View>;

}  // namespace reductions
}  // namespace zipper::views
#endif
