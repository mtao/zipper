#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_DETAIL_SWAP_PARITY_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_DETAIL_SWAP_PARITY_HPP

#include <algorithm>
#include <array>
#include <span>
#include <vector>

#include "zipper/types.hpp"

namespace zipper::expression::reductions::detail {
// returns true if even
template <size_t size>
bool swap_parity(std::span<index_type, size> s) {
    using dat =
        std::conditional_t<size == std::dynamic_extent, std::vector<index_type>,
                           std::array<index_type, size>>;

    bool r = true;
    dat a;
    if constexpr (size == std::dynamic_extent) {
        a.resize(s.size());
    }
    std::ranges::copy(s.begin(), s.end(), a.begin());
    for (size_t j = 0; j < a.size(); ++j) {
        while (a[j] != j) {
            std::swap(a[j], a[a[j]]);
            r = !r;
        }
    }
    return r;
}
}  // namespace zipper::expression::reductions::detail
#endif
