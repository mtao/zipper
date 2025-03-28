#if !defined(ZIPPER_VIEWS_DIMENSIONEDVIEWBASE_HPP)
#define ZIPPER_VIEWS_DIMENSIONEDVIEWBASE_HPP

#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/views/detail/ViewTraits.hpp"
#include "StaticViewBase.hpp"
#include "DynamicViewBase.hpp"
namespace zipper::views {
template <typename Derived>
using DimensionedViewBase = std::conditional_t<
    zipper::detail::ExtentsTraits<typename detail::ViewTraits<Derived>::extents_type>::is_static,
    StaticViewBase<Derived>, DynamicViewBase<Derived>>;
}
#endif
