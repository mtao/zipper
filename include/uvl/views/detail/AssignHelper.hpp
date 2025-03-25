#if !defined(UVL_VIEWS_DETAIL_ASSIGN_HELPER_HPP)
#define UVL_VIEWS_DETAIL_ASSIGN_HELPER_HPP
#include "ViewTraits.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/detail/ExtentsTraits.hpp"
#include "uvl/detail/extents/all_extents_indices.hpp"
namespace uvl::storage {
template <typename ValueType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
class PlainObjectStorage;
}
namespace uvl::views::detail {
template <concepts::ViewDerived From, concepts::ViewDerived To>
struct AssignHelper {
    using to_extents_type = typename To::extents_type;
    using from_extents_type = typename To::extents_type;
    using value_type = typename To::value_type;
    using extents_type = to_extents_type;

    using traits = To::traits;

    using layout_policy = uvl::default_layout_policy;
    using accessor_policy = uvl::default_accessor_policy<value_type>;
    // using layout_policy = traits::layout_policy;
    // using accessor_policy = traits::accessor_policy;
    // using value_accessor_type = traits::value_accessor_type;

    using to_extents_traits =
        uvl::detail::ExtentsTraits<typename To::extents_type>;
    using from_extents_traits =
        uvl::detail::ExtentsTraits<typename To::extents_type>;
    static_assert(
        to_extents_traits::template is_convertable_from<from_extents_type>());

    static void assign_direct(const From& from, To& to) {
        if constexpr (extents_type::rank() == 0) {
            to() = from();
        } else {
            for (const auto& i :
                 uvl::detail::extents::all_extents_indices(to.extents())) {
                to(i) = from(i);
            }
        }
    }

    static void assign(const From& from, To& to) {
        using VTraits = detail::ViewTraits<From>;
        constexpr static bool assigning_from_infinite =
            VTraits::extents_type::rank() == 0;
        constexpr static bool should_resize =
            !assigning_from_infinite && to_extents_traits::is_dynamic;
        if constexpr (VTraits::is_coefficient_consistent) {
            if constexpr (should_resize) {
                to.resize(from.extents());
            }
            assign_direct(from, to);
        } else {
            using POS =
                storage::PlainObjectStorage<value_type, extents_type,
                                            layout_policy, accessor_policy>;
            POS pos(to_extents_traits::convert_from(from.extents()));

            AssignHelper<From, POS>::assign_direct(from, pos);
            if constexpr (should_resize) {
                to.resize(from.extents());
            }
            AssignHelper<POS, To>::assign_direct(pos, to);
        }
    }
};

}  // namespace uvl::views::detail
#endif
