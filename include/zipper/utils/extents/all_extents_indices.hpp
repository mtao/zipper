#if !defined(ZIPPER_UTILS_EXTENTS_ALL_EXTENT_INDICES_HPP)
#define ZIPPER_UTILS_EXTENTS_ALL_EXTENT_INDICES_HPP

#include <version>

#include "zipper/types.hpp"

// Use std::ranges::views::cartesian_product when available (GCC 13+, MSVC
// 17.7+).  Apple Clang's libc++ does not yet provide it, so we fall back to
// a hand-rolled counter-based range that produces the same
// std::tuple<index_type, ...> values in row-major order.
#if defined(__cpp_lib_ranges_cartesian_product) \
    && __cpp_lib_ranges_cartesian_product >= 202207L
#define ZIPPER_HAS_CARTESIAN_PRODUCT 1
#include <ranges>
#else
#define ZIPPER_HAS_CARTESIAN_PRODUCT 0
#include <array>
#include <cstddef>
#include <tuple>
#endif

namespace zipper::utils::extents {

#if !ZIPPER_HAS_CARTESIAN_PRODUCT
namespace detail {

    /// A lightweight range that iterates every index combination of an
    /// N-dimensional extent in row-major order (last dimension varies fastest).
    /// Each dereference yields a std::tuple<index_type, ...> with Rank
    /// elements.
    template <std::size_t Rank>
    class extents_index_range {
        std::array<index_type, Rank> m_extents;
        bool m_empty = false;

      public:
        struct sentinel {};

        class iterator {
            std::array<index_type, Rank> m_idx{};
            const std::array<index_type, Rank> *m_extents = nullptr;
            bool m_at_end = false;

          public:
            using difference_type = std::ptrdiff_t;

            iterator() = default;
            explicit iterator(const std::array<index_type, Rank> *extents,
                              bool at_end)
              : m_extents(extents), m_at_end(at_end) {}

            auto operator*() const {
                return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                    return std::tuple{m_idx[Is]...};
                }(std::make_index_sequence<Rank>{});
            }

            auto operator++() -> iterator & {
                for (std::size_t d = Rank; d-- > 0;) {
                    if (++m_idx[d] < (*m_extents)[d]) return *this;
                    m_idx[d] = 0;
                }
                m_at_end = true;
                return *this;
            }

            auto operator++(int) -> iterator {
                auto tmp = *this;
                ++*this;
                return tmp;
            }

            friend auto operator==(const iterator &it, sentinel) -> bool {
                return it.m_at_end;
            }
        };

        explicit extents_index_range(std::array<index_type, Rank> ext)
          : m_extents(ext) {
            for (auto e : m_extents) {
                if (e == 0) {
                    m_empty = true;
                    break;
                }
            }
        }

        auto begin() const -> iterator { return iterator{&m_extents, m_empty}; }
        auto end() const -> sentinel { return {}; }
    };

} // namespace detail
#endif // !ZIPPER_HAS_CARTESIAN_PRODUCT

template <index_type... SIndices, index_type... Indices>
auto all_extents_indices(const zipper::extents<SIndices...> &extents,
                         std::integer_sequence<index_type, Indices...>) {
#if ZIPPER_HAS_CARTESIAN_PRODUCT
    return std::ranges::views::cartesian_product(
        std::ranges::views::iota(index_type(0), extents.extent(Indices))...);
#else
    return detail::extents_index_range<sizeof...(Indices)>(
        std::array<index_type, sizeof...(Indices)>{extents.extent(Indices)...});
#endif
}

// returns every index available in an extent
template <index_type... indices>
auto all_extents_indices(const zipper::extents<indices...> &extents) {
    return all_extents_indices(
        extents, std::make_integer_sequence<index_type, sizeof...(indices)>{});
}

// returns every index available in an extent (static only)
template <index_type... indices>
auto all_extents_indices() {
    return all_extents_indices(zipper::extents<indices...>{});
}

// get all extents from a set of extent sizes (TODO: set dimensions)
template <typename... Args>
auto all_extents_indices(Args... indices) {
    return all_extents_indices(dextents<sizeof...(Args)>{indices...});
}
} // namespace zipper::utils::extents
#endif
