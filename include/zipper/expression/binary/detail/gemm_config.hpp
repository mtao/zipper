#if !defined(ZIPPER_EXPRESSION_BINARY_DETAIL_GEMM_CONFIG_HPP)
#define ZIPPER_EXPRESSION_BINARY_DETAIL_GEMM_CONFIG_HPP

/// @file gemm_config.hpp
/// @brief Dimensioning of the blocked GEMM kernel (gemm_kernel.hpp): SIMD
/// width, register tile, cache blocking and execution policy.
///
/// Everything here is either compile-time (per TU, overridable with the
/// ZIPPER_GEMM_* macros below) or plain data in GemmConfig. None of it
/// affects results, only speed.

#include <algorithm>
#include <cstddef>
#include <execution>
#include <experimental/simd>
#include <type_traits>

#include "zipper/types.hpp"

namespace zipper::expression::binary::detail::gemm {

using index_type = zipper::index_type;

// parameters for GEMM:
// C = alpha * A * B + beta * C
// C \in R^{mxn}
// A \in R^{mxk}
// B \in R^{kxn}
//
// Tuning gemm can be done via a few platform dimension parameters. This is to
// make it so that things are generic between float/double:
//   ZIPPER_GEMM_VECTOR_BYTES     bytes per SIMD register (16 SSE/NEON,
//                                32 AVX2, 64 AVX-512; default: native)
//   ZIPPER_GEMM_SIMD_REGISTERS   architectural SIMD registers (default: 32
//                                with AVX-512 or on AArch64, else 16)
//   ZIPPER_GEMM_TILE_MR_VECTORS  tile rows, in vectors (default: 2)
//   ZIPPER_GEMM_TILE_NR          tile columns (default: computed from register
//                                budget)
//   ZIPPER_DISABLE_GEMM_KERNEL   route products to the generic path
//
// For per-type control, specialize GemmTuning<T> (visible in every TU that
// multiplies T). Mixing differently-configured TUs in one program is the
// user's responsibility (the kernel's symbols do not encode the tuning).

/// Shape of the register tile: the microkernel computes an mr×nr block of C,
/// holding it in (mr / W)·nr SIMD accumulators.
struct Tile {
    index_type mr;
    index_type nr;
};

/// A tile is usable with W-wide vectors when its rows split into whole
/// vectors.
template <Tile S, index_type W>
concept SimdTile = S.mr > 0 && S.nr > 0 && W > 0 && S.mr % W == 0;

/// SIMD registers the microkernel keeps live for tile S with W-wide vectors.
/// Each k-step it holds
///   - (mr / W) · nr  accumulators (the C tile, live across all of K),
///   - mr / W         vectors of the current column of packed A,
///   - 1              broadcast of the current element of packed B.
consteval auto live_registers(Tile S, index_type W) -> index_type {
    const index_type mr_vectors = S.mr / W;
    return mr_vectors * S.nr + mr_vectors + 1;
}

/// The widest tile (most columns) with mr_vectors-vector columns that fits
/// in `registers`: solve live_registers <= registers for nr.
consteval auto max_nr(index_type registers, index_type mr_vectors)
    -> index_type {
    return (registers - mr_vectors - 1) / mr_vectors;
}

/// Architectural SIMD registers of the target.
inline constexpr index_type simd_registers =
#if defined(ZIPPER_GEMM_SIMD_REGISTERS)
    ZIPPER_GEMM_SIMD_REGISTERS;
#elif defined(__AVX512F__) || defined(__aarch64__)
    32;
#else
    16;
#endif

/// Per-type kernel dimensioning. Specialize to tune one type independently;
/// the primary template derives everything from the type-independent
/// description above. A specialization must provide `simd_width`,
/// `registers` and `tile` (the rest are inputs to the defaults):
///
///   template <> struct zipper::expression::binary::detail::gemm::
///       GemmTuning<float> {
///       static constexpr index_type simd_width = 8;   // AVX2
///       static constexpr index_type registers = 16;
///       static constexpr Tile tile{.mr = 16, .nr = 4};
///   };
template <typename T>
struct GemmTuning {
    /// Bytes per SIMD register.
    static constexpr index_type vector_bytes =
#if defined(ZIPPER_GEMM_VECTOR_BYTES)
        ZIPPER_GEMM_VECTOR_BYTES;
#else
        std::experimental::native_simd<T>::size() * sizeof(T);
#endif
    /// Elements of T per SIMD register (2 doubles under SSE, 4 under AVX2,
    /// 8 under AVX-512; twice that for float).
    static constexpr index_type simd_width =
        std::max<index_type>(1, vector_bytes / sizeof(T));
    static constexpr index_type registers = simd_registers;
    /// Tile rows in vectors. Two maximizes FMAs per loaded element,
    /// mr·nr / (mr + nr), for 16 and 32 registers.
    static constexpr index_type mr_vectors =
#if defined(ZIPPER_GEMM_TILE_MR_VECTORS)
        ZIPPER_GEMM_TILE_MR_VECTORS;
#else
        2;
#endif
    static constexpr index_type nr =
#if defined(ZIPPER_GEMM_TILE_NR)
        ZIPPER_GEMM_TILE_NR;
#else
        max_nr(registers, mr_vectors);
#endif
    /// AVX2 double: 8×6, float: 16×6; AVX-512 double: 16×14; SSE double: 4×6.
    static constexpr Tile tile{.mr = mr_vectors * simd_width, .nr = nr};
};

template <typename T>
inline constexpr index_type simd_width = GemmTuning<T>::simd_width;

/// The register tile used for T unless a caller picks one explicitly.
template <typename T>
inline constexpr Tile default_tile = [] consteval {
    constexpr Tile S = GemmTuning<T>::tile;
    static_assert(SimdTile<S, simd_width<T>>,
                  "GemmTuning: tile rows must be a whole number of vectors");
    static_assert(live_registers(S, simd_width<T>) <= GemmTuning<T>::registers,
                  "GemmTuning: the register tile does not fit in the SIMD "
                  "registers and would spill");
    return S;
}();

// ── Cache blocking ──────────────────────────────────────────────────────
//
// Cache sizes are inputs, not queried: the standard has no portable way to
// ask, so the defaults are a documented assumption (a typical desktop core)
// that a TU can override with the macros below, or a caller with a
// GemmConfig. Block sizes never affect results, only speed.

#if !defined(ZIPPER_GEMM_L1D_BYTES)
#define ZIPPER_GEMM_L1D_BYTES (32 * 1024)
#endif
#if !defined(ZIPPER_GEMM_L2_BYTES)
#define ZIPPER_GEMM_L2_BYTES (512 * 1024)
#endif
#if !defined(ZIPPER_GEMM_L3_BYTES)
// The share of L3 one thread can expect, not the whole (shared) cache.
#define ZIPPER_GEMM_L3_BYTES (4 * 1024 * 1024)
#endif

/// Per-core cache capacities in bytes.
struct CacheSizes {
    std::size_t l1d = ZIPPER_GEMM_L1D_BYTES;
    std::size_t l2 = ZIPPER_GEMM_L2_BYTES;
    std::size_t l3 = ZIPPER_GEMM_L3_BYTES;
};

/// Cache block sizes of the five-loop GEMM: a kc×nr panel of B lives in L1,
/// an mc×kc block of A in L2, and a kc×nc block of B in L3.
struct BlockSizes {
    index_type kc;
    index_type mc;
    index_type nc;
};

/// The BLIS analytical model (Low et al. 2016), simplified: each level holds
/// its operand in half of the cache, leaving room for the streamed one.
template <typename T, Tile S>
constexpr auto blocking_for(CacheSizes c) -> BlockSizes {
    const auto fit = [](std::size_t bytes, index_type per, index_type mult) {
        const auto n = static_cast<index_type>(bytes / 2 / sizeof(T)) / per;
        return std::max(mult, n / mult * mult);
    };
    // L1 streams an mr×kc sliver of A and holds a kc×nr panel of B.
    const index_type kc = fit(c.l1d, S.mr + S.nr, 8);
    return {.kc = kc, .mc = fit(c.l2, kc, S.mr), .nc = fit(c.l3, kc, S.nr)};
}

/// A standard execution policy (std::execution::seq, par, par_unseq, ...).
template <typename P>
concept ExecutionPolicy = std::is_execution_policy_v<std::remove_cvref_t<P>>;

/// Run-time GEMM configuration. The register tile is compile-time (it shapes
/// the microkernel); cache blocking is plain data; the execution policy
/// (default: sequential) parallelizes over row blocks of C when it is not
/// std::execution::seq. With libstdc++, parallel policies need TBB linked.
template <typename T,
          Tile S,
          ExecutionPolicy Policy = std::execution::sequenced_policy>
struct GemmConfig {
    static constexpr Tile tile = S;
    BlockSizes blocks = blocking_for<T, S>(CacheSizes{});
    Policy policy{};
};

} // namespace zipper::expression::binary::detail::gemm

#endif
