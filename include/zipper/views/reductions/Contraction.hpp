#include "zipper/concepts/TupleLike.hpp"
#if !defined(ZIPPER_VIEWS_CONTRACTION_HPP)
#define ZIPPER_VIEWS_CONTRACTION_HPP

#include "zipper/concepts/TupleLike.hpp"
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/detail/extents/all_extents_indices.hpp"
#include "zipper/detail/extents/convert_extents.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/views/detail/ViewTraits.hpp"

namespace zipper::views {
namespace reductions {

namespace detail {
template <typename>
struct folded_extents;

template <index_type... N>
struct folded_extents<extents<N...>> {
    using base_extents_type = extents<N...>;
    constexpr static std::size_t base_rank = sizeof...(N);

    constexpr static std::size_t rank = base_rank / 2;

    template <typename>
    struct half_;
    template <index_type... M>
    struct half_<std::integer_sequence<rank_type, M...>> {
        using extents_type =
            extents<(base_extents_type::static_extent(M) == std::dynamic_extent
                         ? base_extents_type::static_extent(base_rank - M)
                         : base_extents_type::static_extent(M))...>;

        extents_type get_extents(const base_extents_type& e) {
            return zipper::detail::extents::convert_extents<base_extents_type,
                                                         M...>(e);
        }
    };

    using half =
        half_<decltype(std::make_integer_sequence<rank_type, base_rank / 2>{})>;
    using extents_type = half::extents_type;

    template <rank_type M, concepts::TupleLike tuple_type>
    auto get_arg(const tuple_type& t) {
        constexpr static rank_type I = M >= rank ? base_rank - M - 1 : M;
        return std::get<I>(t);
    }
};

}  // namespace detail

template <concepts::ViewDerived View>
class Contraction {
   public:
    using self_type = Contraction<View>;
    using view_type = View;
    using view_traits = zipper::views::detail::ViewTraits<view_type>;
    using value_type = typename View::value_type;
    using base_extents_type = View::extents_type;
    using folder = detail::folded_extents<base_extents_type>;
    using extents_type = folder::extents_type;

    Contraction(View&& v) : m_view(v) {}
    Contraction(const View& v) : m_view(v) {}

    Contraction(Contraction&& v) = default;
    Contraction(const Contraction& v) = default;
    Contraction& operator=(Contraction&& v) = delete;
    Contraction& operator=(const Contraction& v) = delete;

    template <rank_type... N>
    value_type coeff_(concepts::TupleLike auto const& t,
                      std::integer_sequence<rank_type, N...>) {
        return m_view(folder::get_arg<N>(t)...);
    }

    value_type operator()() const {
        value_type v = 0.0;
        for (const auto& i :
             zipper::detail::extents::all_extents_indices(m_extents)) {
            v += coeff_(
                i,
                std::make_integer_sequence<rank_type, extents_type::rank()>{});
        }
        return v;
    }

   private:
    const View& m_view;
    extents_type m_extents;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

template <concepts::ViewDerived View>
Contraction(const View&) -> Contraction<View>;

}  // namespace reductions
}  // namespace zipper::views
#endif
