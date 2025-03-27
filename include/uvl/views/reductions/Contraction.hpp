#include "uvl/concepts/TupleLike.hpp"
#if !defined(UVL_VIEWS_CONTRACTION_HPP)
#define UVL_VIEWS_CONTRACTION_HPP

#include "uvl/concepts/TupleLike.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/detail/extents/all_extents_indices.hpp"
#include "uvl/detail/extents/convert_extents.hpp"
#include "uvl/detail/pack_index.hpp"
#include "uvl/views/detail/ViewTraits.hpp"

namespace uvl::views {
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
            return uvl::detail::extents::convert_extents<base_extents_type,
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
    using view_traits = uvl::views::detail::ViewTraits<view_type>;
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
             uvl::detail::extents::all_extents_indices(m_extents)) {
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
}  // namespace uvl::views
#endif
