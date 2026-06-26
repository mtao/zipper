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

#include "zipper/expression/concepts/capabilities.hpp"
#include "zipper/storage/layout_types.hpp"
#include "zipper/types.hpp"

namespace zipper::expression::binary::detail::gemm {

using index_type = zipper::index_type;

// Lightweight transpose adaptor for a source operand: presents e(j, i) as
// (i, j) with swapped extents. Used to route a column-major target through the
// row-major kernel (see gemm() below). Intentionally minimal — it only exposes
// the element accessor and extents the packing/dispatch path needs, and is NOT
// a zipper Expression, so `LinearArray<Transposed>` is false and packing uses
// the (layout-agnostic) element-accessor path. The microkernel is untouched, so
// the column-major case keeps the same unit-stride accumulator stores.
template <typename Expr>
struct Transposed {
    const Expr& e;
    constexpr auto operator()(index_type i, index_type j) const {
        return e(j, i);
    }
    constexpr index_type extent(rank_type d) const {
        return e.extent(d == 0 ? 1 : 0);
    }
};
template <typename Expr>
Transposed(const Expr&) -> Transposed<Expr>;

// The fast packing path applies to any operand that is a `LinearArray`
// (`zipper::expression::concepts::LinearArray`): it has a layout `mapping()`
// (per-dimension strides) AND a flat unchecked `operator[]` over its buffer, so
// packing linearizes as `operand[ i*s0 + j*s1 ]`. This holds for dense storage
// of any layout (row- or column-major) and, as the linearization tickets land,
// for slices and transposes too. Operands that aren't linear arrays
// (value-computing / lazy) use the general expression-accessor fallback below.
using zipper::expression::concepts::LinearArray;

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

inline constexpr index_type MC = 64;
inline constexpr index_type KC = 256;
inline constexpr index_type NC = 256;
// 4×4 keeps all accumulators in registers (4 YMM for doubles). Wider tiles
// spill with this auto-vectorized formulation; step (2) replaces it with a
// hand-vectorized microkernel.
inline constexpr index_type MR = 4;
inline constexpr index_type NR = 4;

inline constexpr index_type round_up(index_type x, index_type m) {
    return ((x + m - 1) / m) * m;
}

// Pack an mc×kc block of A read via the expression accessor into MR-tall
// panels (rows past mc zero-padded): Apack[panel*(MR*kc)+p*MR+ir] = A(i0+.., p0+p)
template <typename AExpr, typename T>
void pack_A(const AExpr& A, index_type i0, index_type p0, index_type mc,
            index_type kc, T* Apack) {
    const index_type panels = round_up(mc, MR) / MR;
    for (index_type panel = 0; panel < panels; ++panel) {
        T* dst = Apack + static_cast<std::size_t>(panel) * MR * kc;
        for (index_type ir = 0; ir < MR; ++ir) {
            const index_type i = panel * MR + ir;
            if (i >= mc) {
                for (index_type p = 0; p < kc; ++p) dst[p * MR + ir] = T(0);
            } else if constexpr (LinearArray<AExpr>) {
                // Lower-order re-indexing through the layout mapping: resolve
                // the row base once via the row stride, then step along the
                // column stride (== 1 for row-major → sequential), reading the
                // buffer through the unchecked flat operator[].
                const index_type s0 = A.mapping().stride(0),
                                 s1 = A.mapping().stride(1);
                const index_type base =
                    static_cast<index_type>(i0 + i) * s0 + p0 * s1;
                for (index_type p = 0; p < kc; ++p)
                    dst[p * MR + ir] = A[base + p * s1];
            } else {
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
    const index_type panels = round_up(nc, NR) / NR;
    for (index_type panel = 0; panel < panels; ++panel) {
        T* dst = Bpack + static_cast<std::size_t>(panel) * NR * kc;
        for (index_type p = 0; p < kc; ++p) {
            if constexpr (LinearArray<BExpr>) {
                const index_type s0 = B.mapping().stride(0),
                                 s1 = B.mapping().stride(1);
                const index_type base =
                    static_cast<index_type>(p0 + p) * s0 + j0 * s1;
                for (index_type jr = 0; jr < NR; ++jr) {
                    const index_type j = panel * NR + jr;
                    dst[p * NR + jr] = (j < nc) ? B[base + j * s1] : T(0);
                }
            } else {
                for (index_type jr = 0; jr < NR; ++jr) {
                    const index_type j = panel * NR + jr;
                    dst[p * NR + jr] = (j < nc) ? elem(B, p0 + p, j0 + j) : T(0);
                }
            }
        }
    }
}

// MR×NR microkernel: Ctile(mr×nr, row stride ldc) += packed panels. Raw
// pointers so the accumulators stay in registers and the j-loop vectorizes;
// see the file header on why this must not be a bounds-checked view here.
template <typename T>
void microkernel(index_type kc, const T* Apanel, const T* Bpanel, T* Ctile,
                 index_type ldc, index_type mr, index_type nr) {
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

// Public entry: C = A * B, with A,B zipper expressions and C the concrete
// target (row- OR column-major). Aliasing guard: if an operand that owns a
// buffer aliases C's buffer (e.g. in-place A = A*B), compute into a temporary
// then copy out.
//
// Layout-generic write: the blocked kernel always emits a ROW-MAJOR M'×N'
// result into a contiguous buffer. A column-major C(M×N) occupies the same
// bytes as a row-major Cᵀ(N×M), and Cᵀ = (A·B)ᵀ = Bᵀ·Aᵀ. So for a column-major
// target we run the kernel on the transposed operands with M and N swapped; the
// row-major store lands exactly on the column-major layout. No microkernel
// change, no privileged layout.
template <typename AExpr, typename BExpr, typename CExpr>
void gemm(const AExpr& A, const BExpr& B, CExpr& C) {
    using T = std::remove_cv_t<std::remove_pointer_t<decltype(C.data())>>;
    const index_type M = A.extent(0), K = A.extent(1), N = B.extent(1);
    T* const Cd = C.data();
    constexpr bool c_col_major =
        std::is_same_v<typename CExpr::layout_policy, zipper::storage::layout_left>;

    // Run the product into a row-major contiguous buffer `out`, transposing for
    // a column-major target so the same bytes spell the right layout.
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
}

}  // namespace zipper::expression::binary::detail::gemm

#endif
