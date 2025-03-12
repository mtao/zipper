#if !defined(UVL_VIEWS_MAPPEDVIEWBASE_HPP)
#define UVL_VIEWS_MAPPEDVIEWBASE_HPP

#include "uvl/detail/ExtentsTraits.hpp"
#include "uvl/views/detail/ViewTraits.hpp"
#include "StaticMappedViewBase.hpp"
#include "DynamicMappedViewBase.hpp"
namespace uvl::views {
template <typename Derived>
using MappedViewBase = std::conditional_t<
    uvl::detail::ExtentsTraits<typename detail::ViewTraits<Derived>::extents_type>::is_static,
    StaticMappedViewBase<Derived>, DynamicMappedViewBase<Derived>>;
}
#endif
