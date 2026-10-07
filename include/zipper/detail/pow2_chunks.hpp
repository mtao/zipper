#pragma once

/// @file pow2_chunks.hpp
/// @brief Cover a range with power-of-two-sized chunks, for SIMD loops.
///
/// for_each_pow2_chunk<Width>(n, f) calls `f.template operator()<V>(i)` for
/// consecutive chunks [i, i + V) covering [0, n): as many chunks of Width as
/// fit, then each narrower power of two (Width/2, Width/4, ..., 1) at most
/// once -- the binary digits of the remainder. With Width the SIMD width,
/// every element is handled by a vector of compile-time size V, without a
/// scalar remainder loop or masking: a remainder of up to 15 floats under
/// AVX-512 takes vectors of 8 + 4 + 2 + 1.

#include <bit>

#include "zipper/types.hpp"

namespace zipper::detail {

namespace pow2_chunks_detail {
    template <index_type V, typename F>
    constexpr void tail(index_type i, index_type n, F &f) {
        if constexpr (V >= 1) {
            if (n - i >= V) {
                f.template operator()<V>(i);
                i += V;
            }
            tail<V / 2>(i, n, f);
        }
    }
} // namespace pow2_chunks_detail

template <index_type Width, typename F>
    requires(std::has_single_bit(Width))
constexpr void for_each_pow2_chunk(index_type n, F &&f) {
    index_type i = 0;
    for (; i + Width <= n; i += Width) { f.template operator()<Width>(i); }
    pow2_chunks_detail::tail<Width / 2>(i, n, f);
}

} // namespace zipper::detail
