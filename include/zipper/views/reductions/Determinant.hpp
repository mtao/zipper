#if !defined(ZIPPER_VIEWS_REDUCTIONS_DETERMINANT_HPP)
#define ZIPPER_VIEWS_REDUCTIONS_DETERMINANT_HPP

#include <ranges>
#include <type_traits>

#include "detail/swap_parity.hpp"
#include "zipper/concepts/TupleLike.hpp"
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/detail/extents/all_extents_indices.hpp"
#include "zipper/detail/extents/convert_extents.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/views/detail/ViewTraits.hpp"

namespace zipper::views {
namespace reductions {
namespace detail {

template <typename T>
T det2(const T& a, const T& b, const T& c, const T& d) {
    return a * d - b * c;
}
}  // namespace detail

template <concepts::ViewDerived View>
    requires(View::extents_traits::rank == 2)
class Determinant {
   public:
    using self_type = Determinant<View>;
    using view_type = View;
    using view_traits = zipper::views::detail::ViewTraits<view_type>;
    using value_type = typename View::value_type;
    using base_extents_type = View::extents_type;
    using base_extents_traits =
        zipper::detail::ExtentsTraits<base_extents_type>;
    constexpr static index_type static_rows =
        base_extents_type::static_extent(0);
    constexpr static index_type static_cols =
        base_extents_type::static_extent(1);

    constexpr static size_t size =
        static_rows == std::dynamic_extent ? static_cols : static_rows;

    Determinant(View&& v) : m_view(v) {}
    Determinant(const View& v) : m_view(v) {}

    Determinant(Determinant&& v) = default;
    Determinant(const Determinant& v) = default;
    Determinant& operator=(Determinant&& v) = delete;
    Determinant& operator=(const Determinant& v) = delete;

    template <zipper::concepts::IndexLike... Indices>

    value_type operator()() const {
        if constexpr (base_extents_traits::is_dynamic) {
            assert(m_view.extent(0) == m_view.extent(1));
        } else {
            static_assert(static_rows == static_cols);
        }
        if constexpr (size == 1) {
            return m_view(0, 0);
        } else if constexpr (size == 2) {
            return det2();
        } else if constexpr (size == 3) {
            return det3();
        } else {
            return naive_impl();
        }
    }

    value_type det2() const {
        return detail::det2(m_view(0, 0), m_view(0, 1), m_view(1, 0), m_view(1, 1));
    }
    // borrowed from eigen determinant
    value_type det3() const {
        auto helper = [](const auto& v, index_type a, index_type b,
                         index_type c) {
            return v(0, a) * detail::det2(v(1, b), v(1, c), v(2, b), v(2, c));
        };

        return helper(m_view, 0, 1, 2) - helper(m_view, 1, 0, 2) +
               helper(m_view, 2, 0, 1);
    }
    value_type naive_impl() const {
        value_type v = 0.0;
        using dat = typename std::conditional_t<size == std::dynamic_extent,
                                                std::vector<index_type>,
                                                std::array<index_type, size>>;
        auto io = std::ranges::views::iota(index_type(0), m_view.extent(0));
        dat p;
        if constexpr (size == std::dynamic_extent) {
            p.resize(m_view.extent(0));
        }
        std::ranges::copy(io.begin(), io.end(), p.begin());

        do {
            value_type x = detail::swap_parity<size>(p) ? 1 : -1;
            for (index_type j = 0; j < m_view.extent(0); ++j) {
                const value_type& coeff = m_view(j, p[j]);
                if (coeff == value_type(0)) {
                    x = 0;
                    break;
                }
                x *= coeff;
            }
            v += x;
        } while (std::ranges::next_permutation(p.begin(), p.end()).found);

        return v;
    }

   private:
    const View& m_view;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

template <concepts::ViewDerived View>
Determinant(const View&) -> Determinant<View>;

}  // namespace reductions
}  // namespace zipper::views
#endif
