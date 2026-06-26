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
/// The contiguous packed scratch and the target buffer use raw pointers: the
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
#include <type_traits>
#include <vector>
#if defined(__linux__)
#include <unistd.h>  // sysconf — runtime cache-size query for auto-tuning
#endif

#include "zipper/expression/binary/detail/gemm_eligible.hpp"
#include "zipper/expression/concepts/capabilities.hpp"
#include "zipper/storage/layout_types.hpp"
#include "zipper/types.hpp"

namespace zipper::expression::binary::detail::gemm {

using index_type = zipper::index_type;

// Lightweight transpose adaptor for a source operand: presents e(j, i) as
// (i, j) with swapped extents. Used to route a column-major target through the
// row-major kernel (see gemm() below). The microkernel is untouched, so the
// column-major case keeps the same unit-stride accumulator stores.
//
// A transpose of a buffer-backed operand is the SAME buffer with swapped
// strides, so when the underlying operand is itself fast-packable we expose a
// swapped-stride mapping() + a forwarding operator[]. That lets the packing
// fast path read the buffer directly (operand[base + p*stride]) for the
// transpose-routed column-major case too, instead of falling back to the
// per-element operator() — closing most of the column-major packing overhead.
// The mapping()/operator[] members are CONSTRAINED on the underlying operand
// providing them, so Transposed of a lazy/value-computing operand simply lacks
// them and packing uses the element-accessor fallback (still correct).
template <typename Expr>
struct Transposed {
    const Expr& e;
    constexpr auto operator()(index_type i, index_type j) const {
        return e(j, i);
    }
    constexpr index_type extent(rank_type d) const {
        return e.extent(d == 0 ? 1 : 0);
    }
    constexpr decltype(auto) operator[](index_type k) const
        requires requires(const Expr& x) { x[k]; }
    {
        return e[k];  // same buffer; pack composes the swapped strides below
    }
    constexpr auto data() const
        requires requires(const Expr& x) { x.data(); }
    {
        return e.data();  // a transpose shares the child's buffer
    }
    constexpr auto mapping() const
        requires requires(const Expr& x) { x.mapping().stride(0); }
    {
        // Expose only what the packing path consumes: stride(0)/stride(1),
        // swapped relative to the child (transpose permutes the two strides).
        struct SwappedMapping {
            index_type s0, s1;
            constexpr index_type stride(rank_type d) const {
                return d == 0 ? s0 : s1;
            }
        };
        return SwappedMapping{e.mapping().stride(1), e.mapping().stride(0)};
    }
};
template <typename Expr>
Transposed(const Expr&) -> Transposed<Expr>;

// A pack operand is "fast-packable" when it exposes a layout mapping (per-dim
// strides) AND a flat unchecked operator[] over a buffer, so packing reads it
// as operand[base + p*stride] instead of the per-element operator() path. Real
// operands satisfy this exactly when they are `LinearArray`s (the trait
// guarantees both members); the kernel-internal `Transposed` adaptor satisfies
// it structurally when its underlying operand does. We gate the pack on this
// structural form rather than the LinearArray *trait* precisely so the internal
// adaptor is recognized too — this is a local packing optimization (the
// element-accessor fallback is always correct), not a public capability.
using zipper::expression::concepts::LinearArray;
template <typename E>
concept FastPackable = requires(const std::remove_cvref_t<E>& e, index_type k) {
    e.data();             // raw buffer base (extracted once by the pack)
    e.mapping().stride(0);  // per-dimension strides
    e[k];                 // unchecked flat element access
};

/// Read element (i, j) of an operand expression (general path). Heavy compared
/// to a linearized load — the deep access_pack→coeff→mapping chain — so it's
/// used only when the operand has no layout mapping (truly lazy nodes).
template <typename Expr>
constexpr auto elem(const Expr& e, index_type i, index_type j) {
    return e(i, j);
}

// ── Tier 0: cache-friendly i-k-j reorder (small sizes) ─────────────────
// C (raw, row stride N) assumed zeroed. Broadcast A(i,k) across row k of B.
template <typename AExpr, typename BExpr, typename T>
void gemm_ikj(const AExpr& A, const BExpr& B, T* C, index_type M, index_type N,
              index_type K) {
    for (index_type i = 0; i < M; ++i) {
        T* c_row = C + i * N;
        for (index_type k = 0; k < K; ++k) {
            const T aik = elem(A, i, k);
            for (index_type j = 0; j < N; ++j) c_row[j] += aik * elem(B, k, j);
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
template <typename T>
consteval index_type reg_NR() {
    return simd_lanes<T>();
}
/// Number of accumulator registers (small enough to leave room for operands).
template <typename T>
consteval index_type reg_MR() {
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

/// Auto-tune the cache blocks from queried cache sizes (Goto/BLIS model). Each
/// level is given ~half its capacity (room for the other operand + reuse).
template <typename T>
inline BlockSizes compute_block_sizes() {
    constexpr index_type MR = reg_MR<T>(), NR = reg_NR<T>();
    constexpr std::size_t es = sizeof(T);
    const std::size_t L1 = cache_bytes(1), L2 = cache_bytes(2),
                      L3 = cache_bytes(3);
    auto floor_to = [](index_type v, index_type m) {
        return std::max<index_type>(m, (v / m) * m);
    };
    index_type KC = L1 ? floor_to(static_cast<index_type>(
                                      L1 / (2 * (MR + NR) * es)), 8)
                       : 256;
    KC = std::clamp<index_type>(KC, 32, 512);
    index_type MC = L2 ? floor_to(static_cast<index_type>(L2 / (2 * KC * es)), MR)
                       : 64;
    MC = std::clamp<index_type>(MC, MR, 1024);
    index_type NC = L3 ? floor_to(static_cast<index_type>(L3 / (2 * KC * es)), NR)
                       : 1024;
    NC = std::clamp<index_type>(NC, NR, 8192);
    return {MC, KC, NC};
}

/// Cache blocks for T, computed once on first use.
template <typename T>
inline const BlockSizes& block_sizes() {
    static const BlockSizes bs = compute_block_sizes<T>();
    return bs;
}

// Pack an mc×kc block of A read via the expression accessor into MR-tall
// panels (rows past mc zero-padded): Apack[panel*(MR*kc)+p*MR+ir] = A(i0+.., p0+p)
template <typename AExpr, typename T>
void pack_A(const AExpr& A, index_type i0, index_type p0, index_type mc,
            index_type kc, T* Apack) {
    constexpr index_type MR = reg_MR<T>();
    const index_type panels = round_up(mc, MR) / MR;
    if constexpr (FastPackable<AExpr>) {
        // Extract the buffer base pointer ONCE; then raw-index the hot loop.
        // Per-element operator[] does not vectorize — it must be a raw strided
        // load. Lower-order re-indexing: the row base via the row stride, then
        // the column stride (== 1 for row-major). FastPackable (not LinearArray)
        // is the gate so the internal Transposed adaptor — which forwards
        // data()/mapping() with swapped strides — also takes this fast path on
        // the column-major route, instead of falling back to operator().
        const T* abase = A.data();
        const index_type s0 = A.mapping().stride(0),
                         s1 = A.mapping().stride(1);
        for (index_type panel = 0; panel < panels; ++panel) {
            T* dst = Apack + static_cast<std::size_t>(panel) * MR * kc;
            for (index_type ir = 0; ir < MR; ++ir) {
                const index_type i = panel * MR + ir;
                if (i >= mc) {
                    for (index_type p = 0; p < kc; ++p) dst[p * MR + ir] = T(0);
                } else {
                    const index_type rb = (i0 + i) * s0 + p0 * s1;
                    for (index_type p = 0; p < kc; ++p)
                        dst[p * MR + ir] = abase[rb + p * s1];
                }
            }
        }
    } else {
        for (index_type panel = 0; panel < panels; ++panel) {
            T* dst = Apack + static_cast<std::size_t>(panel) * MR * kc;
            for (index_type ir = 0; ir < MR; ++ir) {
                const index_type i = panel * MR + ir;
                if (i >= mc)
                    for (index_type p = 0; p < kc; ++p) dst[p * MR + ir] = T(0);
                else
                    for (index_type p = 0; p < kc; ++p)
                        dst[p * MR + ir] = elem(A, i0 + i, p0 + p);
            }
        }
    }
}

// Pack a kc×nc block of B into NR-wide panels (cols past nc zero-padded):
// Bpack[panel*(NR*kc)+p*NR+jr] = B(p0+p, j0+..)
template <typename BExpr, typename T>
void pack_B(const BExpr& B, index_type p0, index_type j0, index_type kc,
            index_type nc, T* Bpack) {
    constexpr index_type NR = reg_NR<T>();
    const index_type panels = round_up(nc, NR) / NR;
    if constexpr (FastPackable<BExpr>) {
        const T* bbase = B.data();
        const index_type s0 = B.mapping().stride(0),
                         s1 = B.mapping().stride(1);
        for (index_type panel = 0; panel < panels; ++panel) {
            T* dst = Bpack + static_cast<std::size_t>(panel) * NR * kc;
            for (index_type p = 0; p < kc; ++p) {
                const index_type rb = (p0 + p) * s0 + j0 * s1;
                for (index_type jr = 0; jr < NR; ++jr) {
                    const index_type j = panel * NR + jr;
                    dst[p * NR + jr] = (j < nc) ? bbase[rb + j * s1] : T(0);
                }
            }
        }
    } else {
        for (index_type panel = 0; panel < panels; ++panel) {
            T* dst = Bpack + static_cast<std::size_t>(panel) * NR * kc;
            for (index_type p = 0; p < kc; ++p)
                for (index_type jr = 0; jr < NR; ++jr) {
                    const index_type j = panel * NR + jr;
                    dst[p * NR + jr] =
                        (j < nc) ? elem(B, p0 + p, j0 + j) : T(0);
                }
        }
    }
}

// MR×NR microkernel: Ctile(mr×nr, row stride ldc) += packed panels. Raw
// pointers so the accumulators stay in registers and the j-loop vectorizes;
// see the file header on why this must not be a bounds-checked view here.
template <typename T>
[[gnu::always_inline]] inline void microkernel(index_type kc, const T* Apanel,
                                               const T* Bpanel, T* Ctile,
                                               index_type ldc, index_type mr,
                                               index_type nr) {
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

// C (raw, row stride N) assumed zeroed.
template <typename AExpr, typename BExpr, typename T>
void gemm_blocked(const AExpr& A, const BExpr& B, T* C, index_type M,
                  index_type N, index_type K) {
    constexpr index_type MR = reg_MR<T>(), NR = reg_NR<T>();
    const BlockSizes blk = block_sizes<T>();  // auto-tuned from cache sizes
    const index_type MC = blk.MC, KC = blk.KC, NC = blk.NC;
    std::vector<T> Apack(static_cast<std::size_t>(round_up(MC, MR)) * KC);
    std::vector<T> Bpack(static_cast<std::size_t>(round_up(NC, NR)) * KC);

    for (index_type jc = 0; jc < N; jc += NC) {
        const index_type nc = std::min(NC, N - jc);
        for (index_type pc = 0; pc < K; pc += KC) {
            const index_type kc = std::min(KC, K - pc);
            pack_B(B, pc, jc, kc, nc, Bpack.data());
            for (index_type ic = 0; ic < M; ic += MC) {
                const index_type mc = std::min(MC, M - ic);
                pack_A(A, ic, pc, mc, kc, Apack.data());
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
                gemm_dispatch(Transposed{B}, Transposed{A}, out, N, M, K);
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
        // PATH 2 — general writable target: kernel into row-major scratch, then
        // scatter through the target's own operator() (resolves any layout).
        static thread_local std::vector<T> cbuf;
        const std::size_t need = static_cast<std::size_t>(M) * N;
        if (cbuf.size() < need) cbuf.resize(need);
        gemm_dispatch(A, B, cbuf.data(), M, N, K);  // gemm_dispatch zero-fills
        for (index_type i = 0; i < M; ++i)
            for (index_type j = 0; j < N; ++j)
                C(i, j) = cbuf[static_cast<std::size_t>(i) * N + j];
    }
}

}  // namespace zipper::expression::binary::detail::gemm

#endif
