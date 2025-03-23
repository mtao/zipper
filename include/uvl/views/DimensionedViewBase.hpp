#if !defined(UVL_VIEWS_DIMENSIONEDVIEWBASE_HPP)
#define UVL_VIEWS_DIMENSIONEDVIEWBASE_HPP

#include "uvl/detail//ExtentsTraits.hpp"
#include "uvl/views/detail/ViewTraits.hpp"
#include "StaticViewBase.hpp"
#include "DynamicViewBase.hpp"
namespace uvl::views {
template <typename Derived>
using DimensionedViewBase = std::conditional_t<
    uvl::detail::ExtentsTraits<typename detail::ViewTraits<Derived>::extents_type>::is_static,
    StaticViewBase<Derived>, DynamicViewBase<Derived>>;
}
#endif
