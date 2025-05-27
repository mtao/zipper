#if !defined(ZIPPER_VIEWS_NULLARY_IDENTITYVIEW_HPP)
#define ZIPPER_VIEWS_NULLARY_IDENTITYVIEW_HPP

#include <utility>

#include "NullaryViewBase.hpp"
#include "zipper/concepts/TupleLike.hpp"
#include "zipper/views/DimensionedViewBase.hpp"

namespace zipper::views {
namespace nullary {
template <typename T, index_type... Indices>
class IdentityView;

}
template <typename T, index_type... Indices>
struct detail::ViewTraits<nullary::IdentityView<T, Indices...>>
    : public nullary::detail::DefaultNullaryViewTraits<T, Indices...> {
    constexpr static bool is_value_based = false;
};

namespace nullary {
template <typename T, index_type... Indices>
class IdentityView
    : public NullaryViewBase<IdentityView<T, Indices...>, T, Indices...> {
   public:
    using self_type = IdentityView<T, Indices...>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    using value_type = traits::value_type;

    using Base = NullaryViewBase<IdentityView<T, Indices...>, T, Indices...>;
    IdentityView()
        requires(extents_traits::is_static)
    = default;
    IdentityView(const IdentityView&) = default;
    IdentityView(IdentityView&&) = default;
    IdentityView& operator=(const IdentityView&) = default;
    IdentityView& operator=(IdentityView&&) = default;

    IdentityView(const extents_type& e) : Base(e) {}
    template <typename... Args>
    IdentityView(Args&&... args)
        requires(concepts::IndexPackLike<Args...>)
        : IdentityView(extents_type(std::forward<Args>(args)...)) {}

    template <std::size_t... N>
    constexpr static bool _indicesAllSame(concepts::TupleLike auto const& t,
                                          std::index_sequence<N...>) {
        return ((std::get<N>(t) == std::get<0>(t)) && ...);
        //
    }

    constexpr static bool indicesAllSame(concepts::TupleLike auto const& t) {
        return _indicesAllSame(
            t, std::make_index_sequence<
                   std::tuple_size_v<std::decay_t<decltype(t)>>>{});
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

};  // namespace nullarytemplate<typenameA,typenameB>class AdditionView

template <typename T, index_type... Indices>
IdentityView(const T&, const extents<Indices...>&)
    -> IdentityView<T, Indices...>;

}  // namespace nullary
}  // namespace zipper::views
#endif
