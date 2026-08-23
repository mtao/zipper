#if !defined(ZIPPER_EXTERIOR_DETAIL_COMBINATORICS_HPP)
#define ZIPPER_EXTERIOR_DETAIL_COMBINATORICS_HPP

#include <algorithm>
#include <vector>

#include "zipper/detail/assert.hpp"
#include "zipper/types.hpp"

namespace zipper::exterior::detail {

constexpr auto factorial(index_type n) -> index_type {
    index_type result = 1;
    for (index_type j = 2; j <= n; ++j) {
        result *= j;
    }
    return result;
}

constexpr auto binomial(index_type n, index_type k) -> index_type {
    if (k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    k = std::min(k, n - k);
    index_type result = 1;
    for (index_type j = 1; j <= k; ++j) {
        result = (result * (n - k + j)) / j;
    }
    return result;
}

template <typename Container>
constexpr auto has_duplicate_indices(const Container &indices) -> bool {
    for (index_type j = 0; j < indices.size(); ++j) {
        for (index_type k = j + 1; k < indices.size(); ++k) {
            if (indices[j] == indices[k]) {
                return true;
            }
        }
    }
    return false;
}

template <typename Container>
constexpr auto inversion_parity(const Container &indices) -> int {
    int sign = 1;
    for (index_type j = 0; j < indices.size(); ++j) {
        for (index_type k = j + 1; k < indices.size(); ++k) {
            if (indices[k] < indices[j]) {
                sign = -sign;
            }
        }
    }
    return sign;
}

inline auto complement_indices(index_type ambient_dimension,
                               const std::vector<index_type> &sorted_indices)
    -> std::vector<index_type> {
    std::vector<index_type> result;
    result.reserve(ambient_dimension - sorted_indices.size());

    index_type cursor = 0;
    for (index_type j = 0; j < ambient_dimension; ++j) {
        if (cursor < sorted_indices.size() && sorted_indices[cursor] == j) {
            ++cursor;
        } else {
            result.push_back(j);
        }
    }
    return result;
}

template <typename Container>
constexpr auto combination_rank(index_type ambient_dimension,
                                const Container &sorted_indices)
    -> index_type {
    const index_type degree = sorted_indices.size();
    index_type rank = 0;
    index_type previous = 0;

    for (index_type pos = 0; pos < degree; ++pos) {
        const index_type current = sorted_indices[pos];
        ZIPPER_ASSERT(current < ambient_dimension);
        ZIPPER_ASSERT(pos == 0 || sorted_indices[pos - 1] < current);
        for (index_type candidate = previous; candidate < current; ++candidate) {
            rank += binomial(ambient_dimension - candidate - 1,
                             degree - pos - 1);
        }
        previous = current + 1;
    }
    return rank;
}

inline auto combination_unrank(index_type ambient_dimension, index_type degree,
                               index_type rank) -> std::vector<index_type> {
    std::vector<index_type> result(degree);
    index_type next = 0;
    for (index_type pos = 0; pos < degree; ++pos) {
        for (index_type candidate = next; candidate < ambient_dimension;
             ++candidate) {
            const index_type remaining = degree - pos - 1;
            const index_type count = binomial(ambient_dimension - candidate - 1,
                                              remaining);
            if (rank < count) {
                result[pos] = candidate;
                next = candidate + 1;
                break;
            }
            rank -= count;
        }
    }
    return result;
}

} // namespace zipper::exterior::detail

#endif
