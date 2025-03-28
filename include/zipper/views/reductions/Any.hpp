#if !defined(ZIPPER_VIEWS_ANY_HPP)
#define ZIPPER_VIEWS_ANY_HPP

#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/detail/extents/all_extents_indices.hpp"
#include "zipper/views/detail/ViewTraits.hpp"

namespace zipper::views {
namespace reductions {

template <concepts::ViewDerived View>
class Any {
   public:
    using self_type = Any<View>;
    using view_type = View;
    using view_traits = zipper::views::detail::ViewTraits<view_type>;
    using value_type = bool;  // typename View::value_type;

    Any(View&& v) : m_view(v) {}
    Any(const View& v) : m_view(v) {}

    Any(Any&& v) = default;
    Any(const Any& v) = default;
    Any& operator=(Any&& v) = delete;
    Any& operator=(const Any& v) = delete;

    value_type operator()() const {
        for (const auto& i :
             zipper::detail::extents::all_extents_indices(m_view.extents())) {
            if (value_type(m_view(i))) {
                return true;
            }
        }
        return false;
    }

   private:
    const View& m_view;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

template <concepts::ViewDerived View>
Any(const View&) -> Any<View>;

}  // namespace reductions
}  // namespace zipper::views
#endif
