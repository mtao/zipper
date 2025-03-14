#if !defined(UVL_VIEWS_NULLARY_IDENTITYVIEW_HPP)
#define UVL_VIEWS_NULLARY_IDENTITYVIEW_HPP

#include <utility>

#include "uvl/concepts/TupleLike.hpp"
#include "uvl/views/DimensionedViewBase.hpp"

namespace uvl::views {
namespace nullary {
template <typename T, index_type... Indices>
class IdentityView;

}
template <typename T, index_type... Indices>
struct detail::ViewTraits<nullary::IdentityView<T, Indices...>>
: public detail::DefaultViewTraits<T,extents<Indices...>>
{
    constexpr static bool is_coefficient_consistent = true;
};

namespace nullary {
template <typename T, index_type... Indices>
class IdentityView : public DimensionedViewBase<IdentityView<T, Indices...>> {
   public:
    using self_type = IdentityView<T, Indices...>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
    using value_type = traits::value_type;

    IdentityView()
        requires(extents_traits::is_static)
    = default;
    IdentityView(const IdentityView&) = default;
    IdentityView(IdentityView&&) = default;
    IdentityView& operator=(const IdentityView&) = default;
    IdentityView& operator=(IdentityView&&) = default;

    IdentityView(const extents_type& e) : m_extents(e) {}
    using Base = DimensionedViewBase<self_type>;
    using Base::extent;

    constexpr const extents_type& extents() const { return m_extents; }

    template <std::size_t... N>
    constexpr static bool _indicesAllSame(concepts::TupleLike auto const& t,
                                          std::index_sequence<N...>) {
        return ((std::get<N>(t) == std::get<0>(t)) && ...);
        //
    }

    constexpr static bool indicesAllSame(concepts::TupleLike auto const& t) {
        return _indicesAllSame(
            t, std::make_index_sequence<extents_type::rank()>{});
    }

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        if constexpr (sizeof...(Args) == 1 &&
                      (concepts::TupleLike<Args> && ...)) {
            return indicesAllSame(std::forward<Args>(idxs)...) ? 1 : 0;
        } else {
            return indicesAllSame(std::make_tuple(idxs...)) ? 1 : 0;
        }
    }

   private:
    extents_type m_extents;
};  // namespace nullarytemplate<typenameA,typenameB>class AdditionView

template <typename T, index_type... Indices>
IdentityView(const T&, const extents<Indices...>&)
    -> IdentityView<T, Indices...>;

}  // namespace nullary
}  // namespace uvl::views
#endif
