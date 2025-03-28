#if !defined(ZIPPER_VIEWS_MAPPEDVIEWBASE_HPP)
#define ZIPPER_VIEWS_MAPPEDVIEWBASE_HPP

#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/views/detail/ViewTraits.hpp"
#include "StaticMappedViewBase.hpp"
#include "DynamicMappedViewBase.hpp"
namespace zipper::views {
template <typename Derived>
using MappedViewBase = std::conditional_t<
    zipper::detail::ExtentsTraits<typename detail::ViewTraits<Derived>::extents_type>::is_static,
    StaticMappedViewBase<Derived>, DynamicMappedViewBase<Derived>>;
}
#endif
