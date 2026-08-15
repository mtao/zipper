#if !defined(ZIPPER_EXPRESSION_BINARY_DETAIL_GEMM_KERNEL_HPP)
#define ZIPPER_EXPRESSION_BINARY_DETAIL_GEMM_KERNEL_HPP

/// @file gemm_kernel.hpp
/// @brief Prototype BLAS-style dense matrix-multiply, expression-native.
///
/// `gemm(A, B, C)` takes the zipper operand *expressions* for A and B and the
/// concrete target C. The operands are read only through their element accessor
/// `A(i, k)` during packing, so the source need not be contiguous row-major:
/// transposed views, sub-blocks, and lazy nodes (e.g. `2*A`) all feed the same
/// blocked kernel, packed once into contiguous panels. This is the payoff of
/// treating the zipper expression as the view — it carries layout, extents, and
/// value semantics, and packing materializes whatever shape it presents.
///
/// The kernel leans on the expression layer's own assertion-free flat access
/// (`LinearArray`'s unchecked `operator[]` == `data()[k]`) rather than private
/// adaptors: transposition is the library's `Swizzle` expression (which
/// composes a swapped-stride mapping over the same buffer), and the strided
/// writeback is a plain expression assignment through `AssignHelper`. Only the
/// packed panel scratch — a kernel-private storage format (PW-interleaved,
/// zero-padded panels), not a strided tensor layout — is indexed directly; the
/// register-tiled FMA microkernel must stay unchecked (this build hardens the
/// STL via -D_GLIBCXX_ASSERTIONS, which would put bounds-check barriers in the
/// hot loop — measured ~3× slower at -march=native). The expression accessor in
/// packing is O(MK+KN), dominated by the O(MNK) microkernel, so its abstraction
/// cost is amortized away for non-trivial sizes.
///
/// C(M×N) = A(M×K) * B(K×N). The target may be row- OR column-major: the kernel
/// emits a row-major block and routes a column-major target through the
/// transpose identity Cᵀ = Bᵀ·Aᵀ (see gemm()), so neither layout is privileged.
/// gemm() adds a runtime aliasing guard, since MatrixProduct::assign_to bypasses
/// AssignHelper's temporary.
///
/// FUTURE: dispatch on the operands' compile-time traits — static extents
/// (specialize/unroll), known-zero structure (skip packed blocks) — which the
/// expression exposes and this kernel does not yet exploit.

#include <algorithm>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>
#if defined(__linux__)
#include <unistd.h>  // sysconf — runtime cache-size query for auto-tuning
#endif
#if defined(__AVX512F__) || (defined(__AVX2__) && defined(__FMA__))
#include <immintrin.h>  // hand-vectorized FMA microkernel (double only)
#endif

#include "zipper/expression/binary/detail/gemm_eligible.hpp"
#include "zipper/expression/concepts/capabilities.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"
#include "zipper/expression/nullary/LinearLayoutExpression.hpp"
#include "zipper/expression/unary/Swizzle.hpp"
#include "zipper/storage/SpanData.hpp"
#include "zipper/storage/layout_types.hpp"
#include "zipper/types.hpp"

namespace zipper::expression::binary::detail::gemm {

using index_type = zipper::index_type;
using zipper::expression::concepts::LinearArray;

/// Transpose an operand through the library's own Swizzle expression: e(j, i)
/// presented as (i, j) with swapped extents. Used to route a column-major
/// target through the row-major kernel (see gemm() below) and to feed the B
/// operand to the unified pack. When the child is a LinearArray, Swizzle
/// composes a swapped-stride mapping over the SAME buffer and forwards the
/// unchecked flat operator[]/data(), so the packing fast path applies to
/// transposed operands too; lazy children simply fall back to the
/// element-accessor path (still correct).
template <typename Expr>
auto transposed(const Expr& e) {
    return zipper::expression::unary::Swizzle<const Expr&, 1, 0>(e);
}

// ── Tier 0: cache-friendly i-k-j reorder (small sizes) ─────────────────
// C (raw, row stride N) assumed zeroed. Broadcast A(i,k) across row k of B.
template <typename AExpr, typename BExpr, typename T>
void gemm_ikj(const AExpr& A, const BExpr& B, T* C, index_type M, index_type N,
              index_type K) {
    for (index_type i = 0; i < M; ++i) {
        T* c_row = C + i * N;
        for (index_type k = 0; k < K; ++k) {
            const T aik = A(i, k);
            for (index_type j = 0; j < N; ++j) c_row[j] += aik * B(k, j);
        }
    }
}

// ── Tier 1: blocked + packed + register-tiled microkernel ──────────────

// ── Tunable parameters: auto-derived from platform info ────────────────
//
// REGISTER BLOCK  MR × NR  — COMPILE-TIME, from the target SIMD width.
//   The microkernel accumulates an MR×NR tile of C in vector registers. NR is
//   one SIMD register's worth of T, so the inner (j) loop is exactly one packed
//   FMA. This matters: if NR is narrower than the register (e.g. NR=4 doubles
//   on AVX-512, where a register holds 8), the auto-vectorizer gives up and
//   scalarizes the accumulator onto the stack — which is precisely what made
//   the naive 4×4 tile ~3× slow at -march=native. MR is the count of
//   accumulator registers; kept small (4) so the A-broadcast and B operands
//   also stay in registers (MR + 1 + 1 ≤ register file → no spills).
//   `consteval`, because the SIMD width is a compile-time property of -march
//   (__AVX512F__ / __AVX__ / __SSE2__) — see simd_lanes().
//
// CACHE BLOCK  MC / KC / NC  — RUNTIME, auto-tuned from the machine's real
//   L1/L2/L3 sizes (Goto/BLIS model: an MR×KC + NR×KC pair of micro-panels in
//   L1, an MC×KC block of A in L2, a KC×NC panel of B in L3). Cache *capacity*
//   is NOT a compile-time constant — the binary may run on a different CPU than
//   it was built on — so this can't be consteval. We query it at runtime
//   (sysconf; Eigen uses cpuid), compute the blocks once via compute_block_sizes()
//   and cache them. Falls back to sane defaults when the query is unavailable.

template <typename T>
consteval index_type simd_lanes() {
#if defined(__AVX512F__)
    return static_cast<index_type>(64 / sizeof(T));  // 512-bit register
#elif defined(__AVX__)
    return static_cast<index_type>(32 / sizeof(T));  // 256-bit
#elif defined(__SSE2__) || defined(__ARM_NEON)
    return static_cast<index_type>(16 / sizeof(T));  // 128-bit
#else
    return 1;
#endif
}
/// One full SIMD register wide (so the j-loop is one packed FMA, no scalarize).
///
/// The hand-vectorized AVX-512 double kernel uses a 2-register-wide tile
/// (NR = 16 = two zmm columns): broadcasting each A value once and FMA-ing it
/// against BOTH B columns reuses the broadcast and the load ports better than a
/// 1-wide tile, which measured fastest (see ZIPPER_GEMM_NR override for tuning).
template <typename T>
consteval index_type reg_NR() {
#if defined(__AVX512F__)
    if constexpr (std::is_same_v<T, double>) {
#ifdef ZIPPER_GEMM_NR
        return ZIPPER_GEMM_NR;
#else
        return 16;
#endif
    }
#endif
    return simd_lanes<T>();
}
/// Number of accumulator registers (small enough to leave room for operands).
///
/// For the hand-vectorized double microkernels below we want an 8×NR tile: 8
/// independent accumulator-vector chains (one per A row) hide the FMA latency
/// (Zen 4 / Skylake-X: ~4–5 cyc latency, 2 FMA/cyc → need ~8–10 in flight). For
/// other types / ISAs the auto-vectorized fallback keeps the conservative MR=4
/// (MR + B + A-broadcast must stay within the register file with no spill).
template <typename T>
consteval index_type reg_MR() {
#if defined(__AVX512F__) || (defined(__AVX2__) && defined(__FMA__))
    if constexpr (std::is_same_v<T, double>) return 8;
#endif
    return 4;
}

inline constexpr index_type round_up(index_type x, index_type m) {
    return ((x + m - 1) / m) * m;
}

/// L1d/L2/L3 capacity in bytes for `level` ∈ {1,2,3}; 0 when unknown.
inline std::size_t cache_bytes(int level) {
#if defined(__linux__) && defined(_SC_LEVEL1_DCACHE_SIZE)
    const long s = level == 1   ? sysconf(_SC_LEVEL1_DCACHE_SIZE)
                   : level == 2 ? sysconf(_SC_LEVEL2_CACHE_SIZE)
                                : sysconf(_SC_LEVEL3_CACHE_SIZE);
    return s > 0 ? static_cast<std::size_t>(s) : 0;
#else
    (void)level;
    return 0;
#endif
}

struct BlockSizes {
    index_type MC, KC, NC;
};

/// Auto-tune the cache blocks from queried cache sizes (Goto/BLIS model).
///
/// Constants were sweep-tuned against the hand-vectorized microkernel (Zen 4,
/// 8×16 double tile): an MR×KC + NR×KC micro-panel pair resident in L1, an
/// MC×KC block of A in L2, and a KC×NC panel of B reused across the MC loop.
///   KC: the full L1 micro-panel budget L1/((MR+NR)*es) (not halved — the pair
///       is the only thing that must stay L1-hot), floored to a 128 minimum so
///       the k-loop is long enough to amortize the C load/store epilogue.
///   MC: ~half of L2 over the packed A block (MC×KC), keeping room for the B
///       micro-panel; capped at 512 — larger MC measured slower (TLB / L2
///       pressure from the A block evicting the streamed B panel).
///   NC: the B panel (KC×NC) kept tight in L2/L3 rather than maximal in L3;
///       a smaller NC keeps it hot across the MC sweep (measured optimum well
///       below the L3-derived maximum).
template <typename T>
inline BlockSizes compute_block_sizes() {
    constexpr index_type MR = reg_MR<T>(), NR = reg_NR<T>();
    constexpr std::size_t es = sizeof(T);
    const std::size_t L1 = cache_bytes(1), L2 = cache_bytes(2);
    auto floor_to = [](index_type v, index_type m) {
        return std::max<index_type>(m, (v / m) * m);
    };
    index_type KC = L1 ? floor_to(static_cast<index_type>(
                                      L1 / (2 * (MR + NR) * es)), 8)
                       : 256;
    KC = std::clamp<index_type>(KC, 128, 512);
    // A block (MC×KC) takes ~a third of L2, leaving the rest for the streamed B
    // micro-panel and C tile; the remaining 2/3 sizes the L2-resident B panel.
    index_type MC = L2 ? floor_to(static_cast<index_type>(L2 / (3 * KC * es)), MR)
                       : 64;
    MC = std::clamp<index_type>(MC, MR, 512);
    index_type NC = L2 ? floor_to(static_cast<index_type>(L2 / (2 * KC * es)), NR)
                       : 1024;
    NC = std::clamp<index_type>(NC, NR, 4096);
    return {MC, KC, NC};
}

/// Cache blocks for T, computed once on first use.
template <typename T>
inline const BlockSizes& block_sizes() {
    static const BlockSizes bs = compute_block_sizes<T>();
    return bs;
}

// ── Unified operand packing ─────────────────────────────────────────────
//
// Both operands pack into the SAME panel format: a rc×kc block, viewed as
// (panel-dim r) × (depth k), laid out in PW-interleaved panels with the panel
// remainder zero-padded:
//
//     dst[panel*(PW*kc) + p*PW + r] = op(r0 + panel*PW + r, k0 + p)
//
// The A side packs MR-tall row panels of A(i, k) directly; the B side is the
// identical operation on the TRANSPOSE — NR-wide column panels of B(k, j) are
// row panels of Bᵀ(j, k) — so gemm_blocked() feeds `transposed(B)` here and
// pack_A/pack_B collapse into one implementation differing only in PW.
//
// Fast path (LinearArray operands, including Swizzle-transposed ones): hoist
// the buffer base + strides once and read raw strided loads — the per-element
// operator[] chain does not vectorize. The loop nest is chosen at runtime by
// stride: when the k stride is the small one (A-side on a row-major source) we
// walk r-outer/p-inner (sequential reads); when the panel-dim stride is the
// small one (B-side: Bᵀ's dim 0 is B's contiguous dim 1) we walk
// p-outer/r-inner (sequential reads AND sequential panel writes). Lazy
// operands fall back to the element accessor.
template <index_type PW, typename Expr, typename T>
void pack_panels(const Expr& op, index_type r0, index_type k0, index_type rc,
                 index_type kc, T* pack) {
    const index_type panels = round_up(rc, PW) / PW;
    if constexpr (LinearArray<Expr>) {
        const auto* base = op.data();
        const auto map = op.mapping();
        const index_type sr = map.stride(0), sk = map.stride(1);
        for (index_type panel = 0; panel < panels; ++panel) {
            T* dst = pack + static_cast<std::size_t>(panel) * PW * kc;
            const index_type pr0 = panel * PW;
            if (sk <= sr && pr0 + PW <= rc) {
                // k is the fast axis (A side, full panel): sequential reads
                // per row, strided panel writes.
                for (index_type r = 0; r < PW; ++r) {
                    const index_type rb = (r0 + pr0 + r) * sr + k0 * sk;
                    for (index_type p = 0; p < kc; ++p)
                        dst[p * PW + r] = base[rb + p * sk];
                }
            } else {
                // r is the fast axis (B side) or a ragged panel: sequential
                // panel writes, zero-padding the r-remainder inline.
                for (index_type p = 0; p < kc; ++p) {
                    const index_type kb = (k0 + p) * sk + (r0 + pr0) * sr;
                    for (index_type r = 0; r < PW; ++r)
                        dst[p * PW + r] =
                            (pr0 + r < rc) ? base[kb + r * sr] : T(0);
                }
            }
        }
    } else {
        for (index_type panel = 0; panel < panels; ++panel) {
            T* dst = pack + static_cast<std::size_t>(panel) * PW * kc;
            for (index_type r = 0; r < PW; ++r) {
                const index_type i = panel * PW + r;
                if (i >= rc)
                    for (index_type p = 0; p < kc; ++p) dst[p * PW + r] = T(0);
                else
                    for (index_type p = 0; p < kc; ++p)
                        dst[p * PW + r] = op(r0 + i, k0 + p);
            }
        }
    }
}

// Auto-vectorized fallback microkernel: Ctile(mr×nr, row stride ldc) += packed
// panels. Raw pointers so the accumulators stay in registers and the j-loop
// vectorizes; see the file header on why this must not be a bounds-checked view
// here. Used for every non-double type / non-AVX ISA, and as the ragged-edge
// path for the hand-vectorized double kernels (which only own the full tile).
template <typename T>
[[gnu::always_inline]] inline void microkernel_fallback(
    index_type kc, const T* Apanel, const T* Bpanel, T* Ctile, index_type ldc,
    index_type mr, index_type nr) {
    constexpr index_type MR = reg_MR<T>(), NR = reg_NR<T>();
    T acc[MR][NR] = {};
    for (index_type p = 0; p < kc; ++p) {
        const T* a = Apanel + p * MR;
        const T* b = Bpanel + p * NR;
        for (index_type i = 0; i < MR; ++i)
            for (index_type j = 0; j < NR; ++j) acc[i][j] += a[i] * b[j];
    }
    for (index_type i = 0; i < mr; ++i)
        for (index_type j = 0; j < nr; ++j) Ctile[i * ldc + j] += acc[i][j];
}

// Software-prefetch lookahead, in k-steps, for the packed panels (a panel
// element is one tile-row of doubles). Tunable; the packed panels are already
// L1/L2-hot so the kernel is throughput- not latency-bound on the load side and
// the exact distance is not very sensitive (swept 0..64 → within noise).
#ifndef ZIPPER_GEMM_PF
#define ZIPPER_GEMM_PF 16
#endif

#if defined(__AVX512F__)
// Hand-vectorized double microkernel, 8×8 tile (MR=8, NR=8). Eight independent
// zmm accumulators (one per A row) give eight in-flight FMA chains to hide the
// ~4–5 cyc FMA latency on a 2-FMA/cyc core; the auto-vectorizer's serial schedule
// could not (it was latency-bound, plateauing ~8 GFLOP/s regardless of width).
// Each k-step: load one B vector (8 packed doubles), broadcast each of the 8 A
// values, FMA into its accumulator. Prefetch the packed panels ahead. The full
// 8×8 tile is owned here; ragged edges (mr<8 || nr<8) defer to the fallback.
[[gnu::always_inline]] inline void microkernel_avx512_8x8(
    index_type kc, const double* Apanel, const double* Bpanel, double* Ctile,
    index_type ldc) {
    __m512d c0 = _mm512_setzero_pd(), c1 = _mm512_setzero_pd(),
            c2 = _mm512_setzero_pd(), c3 = _mm512_setzero_pd(),
            c4 = _mm512_setzero_pd(), c5 = _mm512_setzero_pd(),
            c6 = _mm512_setzero_pd(), c7 = _mm512_setzero_pd();
    for (index_type p = 0; p < kc; ++p) {
        const double* a = Apanel + p * 8;
        const double* b = Bpanel + p * 8;
        _mm_prefetch(reinterpret_cast<const char*>(b + 8 * ZIPPER_GEMM_PF),
                     _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(a + 8 * ZIPPER_GEMM_PF),
                     _MM_HINT_T0);
        const __m512d vb = _mm512_loadu_pd(b);
        c0 = _mm512_fmadd_pd(_mm512_set1_pd(a[0]), vb, c0);
        c1 = _mm512_fmadd_pd(_mm512_set1_pd(a[1]), vb, c1);
        c2 = _mm512_fmadd_pd(_mm512_set1_pd(a[2]), vb, c2);
        c3 = _mm512_fmadd_pd(_mm512_set1_pd(a[3]), vb, c3);
        c4 = _mm512_fmadd_pd(_mm512_set1_pd(a[4]), vb, c4);
        c5 = _mm512_fmadd_pd(_mm512_set1_pd(a[5]), vb, c5);
        c6 = _mm512_fmadd_pd(_mm512_set1_pd(a[6]), vb, c6);
        c7 = _mm512_fmadd_pd(_mm512_set1_pd(a[7]), vb, c7);
    }
    // C += acc, row by row (Ctile is row-major, stride ldc).
    _mm512_storeu_pd(Ctile + 0 * ldc,
                     _mm512_add_pd(_mm512_loadu_pd(Ctile + 0 * ldc), c0));
    _mm512_storeu_pd(Ctile + 1 * ldc,
                     _mm512_add_pd(_mm512_loadu_pd(Ctile + 1 * ldc), c1));
    _mm512_storeu_pd(Ctile + 2 * ldc,
                     _mm512_add_pd(_mm512_loadu_pd(Ctile + 2 * ldc), c2));
    _mm512_storeu_pd(Ctile + 3 * ldc,
                     _mm512_add_pd(_mm512_loadu_pd(Ctile + 3 * ldc), c3));
    _mm512_storeu_pd(Ctile + 4 * ldc,
                     _mm512_add_pd(_mm512_loadu_pd(Ctile + 4 * ldc), c4));
    _mm512_storeu_pd(Ctile + 5 * ldc,
                     _mm512_add_pd(_mm512_loadu_pd(Ctile + 5 * ldc), c5));
    _mm512_storeu_pd(Ctile + 6 * ldc,
                     _mm512_add_pd(_mm512_loadu_pd(Ctile + 6 * ldc), c6));
    _mm512_storeu_pd(Ctile + 7 * ldc,
                     _mm512_add_pd(_mm512_loadu_pd(Ctile + 7 * ldc), c7));
}

// Hand-vectorized double microkernel, 8×16 tile (MR=8, NR=16 = two zmm cols).
// Sixteen accumulators; each k-step loads two B vectors and broadcasts each of
// the 8 A values once, FMA-ing it into both columns — so the A broadcast and the
// register file are reused across two FMAs, raising the FMA:load ratio. Sixteen
// independent chains keep both FMA pipes saturated. (16 acc + 2 B + 1 bcast = 19
// zmm, within the 32-register file.)
[[gnu::always_inline]] inline void microkernel_avx512_8x16(
    index_type kc, const double* Apanel, const double* Bpanel, double* Ctile,
    index_type ldc) {
    __m512d c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60,
        c61, c70, c71;
    c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 =
        c60 = c61 = c70 = c71 = _mm512_setzero_pd();
    for (index_type p = 0; p < kc; ++p) {
        const double* a = Apanel + p * 8;
        const double* b = Bpanel + p * 16;
        _mm_prefetch(reinterpret_cast<const char*>(b + 16 * ZIPPER_GEMM_PF),
                     _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(a + 8 * ZIPPER_GEMM_PF),
                     _MM_HINT_T0);
        const __m512d b0 = _mm512_loadu_pd(b), b1 = _mm512_loadu_pd(b + 8);
        __m512d t;
        t = _mm512_set1_pd(a[0]); c00 = _mm512_fmadd_pd(t, b0, c00); c01 = _mm512_fmadd_pd(t, b1, c01);
        t = _mm512_set1_pd(a[1]); c10 = _mm512_fmadd_pd(t, b0, c10); c11 = _mm512_fmadd_pd(t, b1, c11);
        t = _mm512_set1_pd(a[2]); c20 = _mm512_fmadd_pd(t, b0, c20); c21 = _mm512_fmadd_pd(t, b1, c21);
        t = _mm512_set1_pd(a[3]); c30 = _mm512_fmadd_pd(t, b0, c30); c31 = _mm512_fmadd_pd(t, b1, c31);
        t = _mm512_set1_pd(a[4]); c40 = _mm512_fmadd_pd(t, b0, c40); c41 = _mm512_fmadd_pd(t, b1, c41);
        t = _mm512_set1_pd(a[5]); c50 = _mm512_fmadd_pd(t, b0, c50); c51 = _mm512_fmadd_pd(t, b1, c51);
        t = _mm512_set1_pd(a[6]); c60 = _mm512_fmadd_pd(t, b0, c60); c61 = _mm512_fmadd_pd(t, b1, c61);
        t = _mm512_set1_pd(a[7]); c70 = _mm512_fmadd_pd(t, b0, c70); c71 = _mm512_fmadd_pd(t, b1, c71);
    }
#define ZIPPER_GEMM_ST2(r, lo, hi)                                          \
    _mm512_storeu_pd(Ctile + (r) * ldc,                                     \
                     _mm512_add_pd(_mm512_loadu_pd(Ctile + (r) * ldc), lo)); \
    _mm512_storeu_pd(                                                       \
        Ctile + (r) * ldc + 8,                                              \
        _mm512_add_pd(_mm512_loadu_pd(Ctile + (r) * ldc + 8), hi))
    ZIPPER_GEMM_ST2(0, c00, c01); ZIPPER_GEMM_ST2(1, c10, c11);
    ZIPPER_GEMM_ST2(2, c20, c21); ZIPPER_GEMM_ST2(3, c30, c31);
    ZIPPER_GEMM_ST2(4, c40, c41); ZIPPER_GEMM_ST2(5, c50, c51);
    ZIPPER_GEMM_ST2(6, c60, c61); ZIPPER_GEMM_ST2(7, c70, c71);
#undef ZIPPER_GEMM_ST2
}
#elif defined(__AVX2__) && defined(__FMA__)
// Hand-vectorized double microkernel, 8×4 tile (MR=8, NR=4 = one ymm). Same
// design: eight independent ymm accumulator chains to hide FMA latency.
[[gnu::always_inline]] inline void microkernel_avx2_8x4(
    index_type kc, const double* Apanel, const double* Bpanel, double* Ctile,
    index_type ldc) {
    __m256d c0 = _mm256_setzero_pd(), c1 = _mm256_setzero_pd(),
            c2 = _mm256_setzero_pd(), c3 = _mm256_setzero_pd(),
            c4 = _mm256_setzero_pd(), c5 = _mm256_setzero_pd(),
            c6 = _mm256_setzero_pd(), c7 = _mm256_setzero_pd();
    for (index_type p = 0; p < kc; ++p) {
        const double* a = Apanel + p * 8;
        const double* b = Bpanel + p * 4;
        _mm_prefetch(reinterpret_cast<const char*>(b + 4 * 24), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(a + 8 * 24), _MM_HINT_T0);
        const __m256d vb = _mm256_loadu_pd(b);
        c0 = _mm256_fmadd_pd(_mm256_set1_pd(a[0]), vb, c0);
        c1 = _mm256_fmadd_pd(_mm256_set1_pd(a[1]), vb, c1);
        c2 = _mm256_fmadd_pd(_mm256_set1_pd(a[2]), vb, c2);
        c3 = _mm256_fmadd_pd(_mm256_set1_pd(a[3]), vb, c3);
        c4 = _mm256_fmadd_pd(_mm256_set1_pd(a[4]), vb, c4);
        c5 = _mm256_fmadd_pd(_mm256_set1_pd(a[5]), vb, c5);
        c6 = _mm256_fmadd_pd(_mm256_set1_pd(a[6]), vb, c6);
        c7 = _mm256_fmadd_pd(_mm256_set1_pd(a[7]), vb, c7);
    }
    _mm256_storeu_pd(Ctile + 0 * ldc,
                     _mm256_add_pd(_mm256_loadu_pd(Ctile + 0 * ldc), c0));
    _mm256_storeu_pd(Ctile + 1 * ldc,
                     _mm256_add_pd(_mm256_loadu_pd(Ctile + 1 * ldc), c1));
    _mm256_storeu_pd(Ctile + 2 * ldc,
                     _mm256_add_pd(_mm256_loadu_pd(Ctile + 2 * ldc), c2));
    _mm256_storeu_pd(Ctile + 3 * ldc,
                     _mm256_add_pd(_mm256_loadu_pd(Ctile + 3 * ldc), c3));
    _mm256_storeu_pd(Ctile + 4 * ldc,
                     _mm256_add_pd(_mm256_loadu_pd(Ctile + 4 * ldc), c4));
    _mm256_storeu_pd(Ctile + 5 * ldc,
                     _mm256_add_pd(_mm256_loadu_pd(Ctile + 5 * ldc), c5));
    _mm256_storeu_pd(Ctile + 6 * ldc,
                     _mm256_add_pd(_mm256_loadu_pd(Ctile + 6 * ldc), c6));
    _mm256_storeu_pd(Ctile + 7 * ldc,
                     _mm256_add_pd(_mm256_loadu_pd(Ctile + 7 * ldc), c7));
}
#endif

// MR×NR microkernel dispatch: routes full double tiles to the hand-vectorized
// intrinsic kernel (above) when available, else to the auto-vectorized
// fallback. Ragged edges always use the fallback.
template <typename T>
[[gnu::always_inline]] inline void microkernel(index_type kc, const T* Apanel,
                                               const T* Bpanel, T* Ctile,
                                               index_type ldc, index_type mr,
                                               index_type nr) {
    [[maybe_unused]] constexpr index_type MR = reg_MR<T>(), NR = reg_NR<T>();
#if defined(__AVX512F__)
    if constexpr (std::is_same_v<T, double> && MR == 8 && NR == 16) {
        if (mr == 8 && nr == 16) {
            microkernel_avx512_8x16(kc, Apanel, Bpanel, Ctile, ldc);
            return;
        }
    } else if constexpr (std::is_same_v<T, double> && MR == 8 && NR == 8) {
        if (mr == 8 && nr == 8) {
            microkernel_avx512_8x8(kc, Apanel, Bpanel, Ctile, ldc);
            return;
        }
    }
#elif defined(__AVX2__) && defined(__FMA__)
    if constexpr (std::is_same_v<T, double> && MR == 8 && NR == 4) {
        if (mr == 8 && nr == 4) {
            microkernel_avx2_8x4(kc, Apanel, Bpanel, Ctile, ldc);
            return;
        }
    }
#endif
    microkernel_fallback<T>(kc, Apanel, Bpanel, Ctile, ldc, mr, nr);
}

// C (raw, row stride N) assumed zeroed.
template <typename AExpr, typename BExpr, typename T>
void gemm_blocked(const AExpr& A, const BExpr& B, T* C, index_type M,
                  index_type N, index_type K) {
    constexpr index_type MR = reg_MR<T>(), NR = reg_NR<T>();
    const BlockSizes blk = block_sizes<T>();  // auto-tuned from cache sizes
    const index_type MC = blk.MC, KC = blk.KC, NC = blk.NC;
    // B packs as row panels of Bᵀ — one pack implementation for both operands.
    const auto Bt = transposed(B);
    // Reused per-thread scratch: the pack buffers have a fixed size for a given
    // T (MC/KC/NC are constant), so allocating+zeroing them on every call is
    // pure overhead — measured a hard cliff at small M/N where a few hundred KB
    // of fresh allocation dominates the (modest) compute. Grow-only, never
    // shrunk; thread_local so concurrent gemm() calls don't share scratch.
    thread_local std::vector<T> Apack, Bpack;
    const std::size_t need_A = static_cast<std::size_t>(round_up(MC, MR)) * KC;
    const std::size_t need_B = static_cast<std::size_t>(round_up(NC, NR)) * KC;
    if (Apack.size() < need_A) Apack.resize(need_A);
    if (Bpack.size() < need_B) Bpack.resize(need_B);

    for (index_type jc = 0; jc < N; jc += NC) {
        const index_type nc = std::min(NC, N - jc);
        for (index_type pc = 0; pc < K; pc += KC) {
            const index_type kc = std::min(KC, K - pc);
            pack_panels<NR>(Bt, jc, pc, nc, kc, Bpack.data());
            for (index_type ic = 0; ic < M; ic += MC) {
                const index_type mc = std::min(MC, M - ic);
                pack_panels<MR>(A, ic, pc, mc, kc, Apack.data());
                for (index_type jr = 0; jr < nc; jr += NR) {
                    const index_type nr = std::min(NR, nc - jr);
                    const T* Bpanel =
                        Bpack.data() + static_cast<std::size_t>(jr / NR) * NR * kc;
                    for (index_type ir = 0; ir < mc; ir += MR) {
                        const index_type mr = std::min(MR, mc - ir);
                        const T* Apanel = Apack.data() +
                                          static_cast<std::size_t>(ir / MR) * MR * kc;
                        T* Ctile = C + (ic + ir) * N + (jc + jr);
                        microkernel(kc, Apanel, Bpanel, Ctile, N, mr, nr);
                    }
                }
            }
        }
    }
}

inline constexpr index_type kBlockedThreshold = 64;

template <typename AExpr, typename BExpr, typename T>
void gemm_dispatch(const AExpr& A, const BExpr& B, T* C, index_type M,
                   index_type N, index_type K) {
    std::fill(C, C + static_cast<std::size_t>(M) * N, T(0));
    if (M >= kBlockedThreshold || N >= kBlockedThreshold ||
        K >= kBlockedThreshold)
        gemm_blocked(A, B, C, M, N, K);
    else
        gemm_ikj(A, B, C, M, N, K);
}

// Public entry: C = A * B, with A,B zipper expressions and C the target.
//
// Two write strategies, chosen with `if constexpr` on the target's capability:
//
//   PATH 1 — contiguous target (DenseContiguousTarget): write the kernel result
//     straight into C's raw buffer via data(). Layout-generic: the blocked
//     kernel emits a ROW-MAJOR M'×N' result; a column-major C(M×N) occupies the
//     same bytes as a row-major Cᵀ(N×M) and Cᵀ = (A·B)ᵀ = Bᵀ·Aᵀ, so for a
//     column-major target we run the kernel on the transposed operands with M
//     and N swapped and the row-major store lands on the column-major layout.
//     An aliasing guard handles in-place A = A*B (MatrixProduct::assign_to
//     bypasses AssignHelper's temporary): if an operand buffer aliases C's, the
//     kernel computes into a temporary then copies out.
//
//   PATH 2 — any other writable dense rank-2 target (strided / sub-block / view,
//     e.g. a Slice with layout_stride): run the kernel into a row-major scratch
//     buffer, then scatter into the target through its own `C(i, j)`, which
//     resolves the address through the target's mapping (any layout lands
//     correctly). No aliasing guard is needed: the kernel consumes A and B fully
//     (into packed panels + scratch) before the scatter touches C, so aliasing
//     targets are safe.
template <typename AExpr, typename BExpr, typename CExpr>
void gemm(const AExpr& A, const BExpr& B, CExpr& C) {
    using T = std::remove_cvref_t<
        typename zipper::expression::detail::ExpressionTraits<
            std::remove_cvref_t<CExpr>>::element_type>;
    const index_type M = A.extent(0), K = A.extent(1), N = B.extent(1);

    if constexpr (zipper::expression::binary::detail::DenseContiguousTarget<
                      CExpr>) {
        // PATH 1 — contiguous buffer, in-place writeback (unchanged).
        T* const Cd = C.data();
        constexpr bool c_col_major =
            std::is_same_v<typename CExpr::layout_policy,
                           zipper::storage::layout_left>;

        // Run the product into a row-major contiguous buffer `out`, transposing
        // for a column-major target so the same bytes spell the right layout.
        auto run = [&](T* out) {
            if constexpr (c_col_major)
                gemm_dispatch(transposed(B), transposed(A), out, N, M, K);
            else
                gemm_dispatch(A, B, out, M, N, K);
        };

        bool alias = false;
        if constexpr (requires { A.data(); })
            alias = alias || static_cast<const void*>(A.data()) == Cd;
        if constexpr (requires { B.data(); })
            alias = alias || static_cast<const void*>(B.data()) == Cd;

        if (alias) {
            std::vector<T> tmp(static_cast<std::size_t>(M) * N);
            run(tmp.data());
            std::copy(tmp.begin(), tmp.end(), Cd);
            return;
        }
        run(Cd);
    } else {
        // PATH 2 — general writable target: kernel into row-major scratch,
        // then write back as a plain expression assignment. The scratch is
        // viewed as a row-major LinearLayoutExpression over the same buffer
        // and AssignHelper routes it into the target through the target's own
        // layout (any mapping lands correctly) — the same machinery every
        // other zipper assignment uses, so improvements there (vectorized
        // linear assignment) apply here for free.
        static thread_local std::vector<T> cbuf;
        const std::size_t need = static_cast<std::size_t>(M) * N;
        if (cbuf.size() < need) cbuf.resize(need);
        gemm_dispatch(A, B, cbuf.data(), M, N, K);  // gemm_dispatch zero-fills

        using scratch_view_type = zipper::expression::nullary::
            LinearLayoutExpression<zipper::storage::SpanData<const T,
                                                             std::dynamic_extent>,
                                   zipper::dextents<2>,
                                   zipper::storage::layout_right>;
        const scratch_view_type scratch(
            zipper::storage::SpanData<const T, std::dynamic_extent>(
                std::span<const T>(cbuf.data(), need)),
            zipper::dextents<2>(M, N));
        zipper::expression::detail::AssignHelper<scratch_view_type,
                                                 CExpr>::assign(scratch, C);
    }
}

}  // namespace zipper::expression::binary::detail::gemm

#endif
