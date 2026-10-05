#if !defined(ZIPPER_EXPRESSION_BINARY_DETAIL_GEMM_KERNEL_HPP)
#define ZIPPER_EXPRESSION_BINARY_DETAIL_GEMM_KERNEL_HPP

/// @file gemm_kernel.hpp
/// @brief Blocked dense matrix-multiply written in terms of zipper itself.
///
/// Built on step 4x4_15 of how-to-optimize-gemm as ported to zipper
/// (how-to-optimize-gemm-zipper), extended to the BLIS five-loop structure:
///
///   * N, K and M are split into nc / kc / mc blocks (GemmConfig; sized from
///     cache capacities by a constexpr model — see gemm_config.hpp);
///   * each kc×nc block of B is packed ONCE (into kc×nr row-major panels) and
///     reused by every mc block of A, which is packed into mr×kc column-major
///     panels. Packing is plain zipper slice assignment, so any operand
///     expression — dense, strided, transposed, lazy (`2*A`) — feeds the
///     same kernel;
///   * an mr×nr microkernel accumulates rank-1 updates `A.col(p) ⊗ B.row(p)`
///     in explicit SIMD registers and returns a `Matrix<T, mr, nr>`, which is
///     added into the target through a slice — so the target can be any
///     writable dense rank-2 expression (row-/column-major, strided view);
///   * the packing buffers persist across calls (grow-only, thread_local).
///
/// Ragged edges zero-pad the packed panels and add only the valid part of
/// the edge tiles.

#include <algorithm>
#include <array>
#include <cstddef>
#include <execution>
#include <thread>
#include <tuple>
#include <vector>
#include <experimental/simd>
#include <ranges>
#include <type_traits>

// Only MatrixBase (which forward-declares Matrix): this header is reached from
// MatrixProduct.hpp, i.e. from inside MatrixBase.hxx, so it must not pull in
// Matrix.hpp. Everything here is a template instantiated after both exist.
#include "zipper/MatrixBase.hpp"
#include "zipper/concepts/Matrix.hpp"
#include "zipper/expression/binary/detail/gemm_config.hpp"
#include "zipper/expression/binary/detail/gemm_eligible.hpp"
#include "zipper/detail/assert.hpp"
#include "zipper/static_scalar.hpp"
#include "zipper/types.hpp"

namespace zipper::expression::binary::detail::gemm {

using index_type = zipper::index_type;

template <typename T, Tile S>
using APanels =
    zipper::Matrix<T, S.mr, dynamic_extent, false>; // mr × (panels·kc)
template <typename T, Tile S>
using BPanels =
    zipper::Matrix<T, dynamic_extent, S.nr, true>; // (panels·kc) × nr

/// Split [0, n) into consecutive (offset, size) blocks of at most b.
constexpr auto blocks(index_type n, index_type b) {
    return std::views::iota(index_type{0}, n) | std::views::chunk(b)
           | std::views::transform([](auto r) -> auto {
                 return std::pair{
                     r.front(), static_cast<index_type>(std::ranges::size(r))};
             });
}

constexpr auto round_up(index_type n, index_type m) -> index_type {
    return (n + m - 1) / m * m;
}

/// The register-tiled microkernel (tutorial's mymul_4x4, generalized):
/// Apanel is an mr×kc column-major panel, Bpanel a kc×nr row-major panel;
/// returns Apanel * Bpanel as a sum of rank-1 updates a ⊗ b.
template <typename T, Tile S, index_type W = simd_width<T>>
    requires SimdTile<S, W>
auto microkernel(zipper::concepts::Matrix auto const &Apanel,
                 zipper::concepts::Matrix auto const &Bpanel)
    -> zipper::Matrix<T, S.mr, S.nr, false> {
    namespace stdx = std::experimental;
    constexpr index_type R = S.mr / W;
    using vec = stdx::fixed_size_simd<T, W>;

    std::array<std::array<vec, R>, S.nr> cv{};
    for (index_type p : std::views::iota(index_type{0}, Apanel.cols())) {
        auto a = Apanel.col(p);
        auto b = Bpanel.row(p);

        std::array<vec, R> av;
#pragma GCC unroll 16
        for (index_type r = 0; r < R; ++r) {
            av[r].copy_from(&a(r * W), stdx::element_aligned);
        }
        // Unroll explicitly: cv[t][r] must use constant indices to stay in
        // registers, and -O2 (unlike -O3) won't fully unroll this early.
#pragma GCC unroll 16
        for (index_type t = 0; t < S.nr; ++t) {
            const vec bv = b(t);
#pragma GCC unroll 16
            for (index_type r = 0; r < R; ++r) { cv[t][r] += av[r] * bv; }
        }
    }

    // Column-major, so each SIMD vector is a contiguous piece of a column.
    zipper::Matrix<T, S.mr, S.nr, false> C(zipper::uninitialized);
#pragma GCC unroll 16
    for (index_type t = 0; t < S.nr; ++t) {
#pragma GCC unroll 16
        for (index_type r = 0; r < R; ++r) {
            cv[t][r].copy_to(&C(r * W, t), stdx::element_aligned);
        }
    }
    return C;
}

/// dst = alpha * src, without a multiply when alpha is statically 1.
template <typename T, typename Alpha>
void assign_scaled(auto &&dst, auto const &src, Alpha alpha) {
    if constexpr (zipper::concepts::StaticScalarOf<Alpha, 1>) {
        dst = src;
    } else {
        dst = static_cast<T>(alpha) * src;
    }
}

/// Pack alpha times the m×kc block `A` into consecutive mr×kc panels of
/// `Apacked`, zero-padding the rows of the last panel past m. Folding alpha
/// in here costs one multiply per element of A, not per FMA.
template <typename T, Tile S, typename Alpha>
void pack_A(zipper::concepts::Matrix auto const &A,
            APanels<T, S> &Apacked,
            Alpha alpha) {
    using mr_t = zipper::static_index_t<S.mr>;
    const index_type kc = A.cols();
    for (auto [i, mr] : blocks(A.rows(), S.mr)) {
        auto panel = Apacked.col_slice(zipper::slice(i / S.mr * kc, kc));
        if (mr == S.mr) {
            assign_scaled<T>(panel, A.row_slice(zipper::slice(i, mr_t{})), alpha);
        } else {
            assign_scaled<T>(panel.row_slice(zipper::slice(0, mr)),
                             A.row_slice(zipper::slice(i, mr)),
                             alpha);
            panel.row_slice(zipper::slice(mr, S.mr - mr)).set_zero();
        }
    }
}

/// Pack the kc×n block `B` into consecutive kc×nr panels of `Bpacked`,
/// zero-padding the columns of the last panel past n.
template <typename T, Tile S>
void pack_B(zipper::concepts::Matrix auto const &B, BPanels<T, S> &Bpacked) {
    using nr_t = zipper::static_index_t<S.nr>;
    const index_type kc = B.rows();
    for (auto [j, nr] : blocks(B.cols(), S.nr)) {
        auto panel = Bpacked.row_slice(zipper::slice(j / S.nr * kc, kc));
        if (nr == S.nr) {
            panel = B.col_slice(zipper::slice(j, nr_t{}));
        } else {
            panel.col_slice(zipper::slice(0, nr)) =
                B.col_slice(zipper::slice(j, nr));
            panel.col_slice(zipper::slice(nr, S.nr - nr)).set_zero();
        }
    }
}

/// tile = beta * tile + c. Static beta = 0 never reads the tile, so garbage
/// or NaN in C does not leak into the result (BLAS semantics).
template <typename T, typename Beta>
void update_tile(auto &&tile, auto const &c, Beta beta) {
    if constexpr (zipper::concepts::StaticScalarOf<Beta, 0>) {
        tile = c;
    } else if constexpr (zipper::concepts::StaticScalarOf<Beta, 1>) {
        tile += c;
    } else {
        tile = static_cast<T>(beta) * tile + c;
    }
}

/// C = beta * C + Apacked * Bpacked for one packed (mc×kc)·(kc×N) block pair
/// (tutorial's InnerKernel), sweeping mr×nr tiles of C.
template <typename T, Tile S, typename Beta>
void inner_kernel(APanels<T, S> const &Apacked,
                  BPanels<T, S> const &Bpacked,
                  index_type kc,
                  zipper::concepts::Matrix auto &&C,
                  Beta beta) {
    using mr_t = zipper::static_index_t<S.mr>;
    using nr_t = zipper::static_index_t<S.nr>;
    for (auto [j, nr] : blocks(C.cols(), S.nr)) {
        auto Bpanel = Bpacked.row_slice(zipper::slice(j / S.nr * kc, kc));
        for (auto [i, mr] : blocks(C.rows(), S.mr)) {
            auto Apanel = Apacked.col_slice(zipper::slice(i / S.mr * kc, kc));
            const auto c = microkernel<T, S>(Apanel, Bpanel);
            if (mr == S.mr && nr == S.nr) {
                update_tile<T>(
                    C.slice(zipper::slice(i, mr_t{}), zipper::slice(j, nr_t{})),
                    c,
                    beta);
            } else {
                update_tile<T>(
                    C.slice(zipper::slice(i, mr), zipper::slice(j, nr)),
                    c.slice(zipper::slice(0, mr), zipper::slice(0, nr)),
                    beta);
            }
        }
    }
}

/// C = beta * C, with static 0 / 1 not reading / not touching C.
template <typename T, typename Beta>
void scale(zipper::concepts::Matrix auto &&C, Beta beta) {
    if constexpr (zipper::concepts::StaticScalarOf<Beta, 0>) {
        C.set_zero();
    } else if constexpr (!zipper::concepts::StaticScalarOf<Beta, 1>) {
        C *= static_cast<T>(beta);
    }
}

/// Grow-only, per-thread packing buffer: always overwritten before it's read,
/// so it is never zero-filled or shrunk.
template <typename Buffer>
auto packing_buffer(index_type rows, index_type cols) -> Buffer & {
    static thread_local Buffer buf(zipper::uninitialized, rows, cols);
    if (buf.rows() < rows || buf.cols() < cols) {
        buf.resize(std::max(rows, buf.rows()), std::max(cols, buf.cols()));
    }
    return buf;
}

/// C = beta * C + alpha * A * B (BLAS gemm) with zipper Matrix views A (M×K),
/// B (K×N), C (M×N), an S-shaped register tile and the five-loop blocking of
/// BLIS:
///
///   jc: nc-wide column blocks of C and B
///     pc: kc-deep blocks of K          → pack B(pc, jc) once (L3)
///       ic: mc-tall row blocks of A, C → pack alpha·A(ic, pc) once (L2)
///         jr, ir: nr×mr register tiles (inner_kernel)
///
/// beta is applied by the first kc block only; later blocks accumulate.
/// alpha and beta are run-time scalars or zipper::cw<V>; run-time
/// values equal to 0 / 1 are dispatched to the static paths once per call.
/// C must not alias A or B.
template <typename T,
          Tile S = default_tile<T>,
          ExecutionPolicy Policy = std::execution::sequenced_policy,
          zipper::concepts::Coefficient<T> Alpha,
          zipper::concepts::Coefficient<T> Beta>
    requires SimdTile<S, simd_width<T>>
void gemm_blocked(Alpha alpha,
                  zipper::concepts::Matrix auto const &A,
                  zipper::concepts::Matrix auto const &B,
                  Beta beta,
                  zipper::concepts::Matrix auto &&C,
                  GemmConfig<T, S, Policy> const &config = {}) {
    // Run-time 0 / 1 take the static paths (beta = 0 must not read C).
    if constexpr (!zipper::concepts::StaticScalar<Beta>) {
        if (static_cast<T>(beta) == T(0)) {
            return gemm_blocked<T, S>(
                alpha, A, B, zipper::cw<0>, C, config);
        }
        if (static_cast<T>(beta) == T(1)) {
            return gemm_blocked<T, S>(
                alpha, A, B, zipper::cw<1>, C, config);
        }
    }
    if constexpr (!zipper::concepts::StaticScalar<Alpha>) {
        if (static_cast<T>(alpha) == T(1)) {
            return gemm_blocked<T, S>(
                zipper::cw<1>, A, B, beta, C, config);
        }
        if (static_cast<T>(alpha) == T(0)) {
            return gemm_blocked<T, S>(
                zipper::cw<0>, A, B, beta, C, config);
        }
    }

    // Nothing to multiply: C = beta * C, without reading A or B.
    if constexpr (zipper::concepts::StaticScalarOf<Alpha, 0>) {
        scale<T>(C, beta);
        return;
    } else {
        if (A.cols() == 0) {
            scale<T>(C, beta);
            return;
        }

        constexpr bool parallel =
            !std::is_same_v<std::remove_cvref_t<Policy>,
                            std::execution::sequenced_policy>;
        const auto [kc_max, nc_max, mc_step] = [&] {
            const auto [kc, mc, nc] = config.blocks;
            if constexpr (parallel) {
                // Enough row blocks to feed every hardware thread.
                const index_type threads = std::max<index_type>(
                    1, std::thread::hardware_concurrency());
                const index_type even =
                    round_up((A.rows() + threads - 1) / threads, S.mr);
                return std::tuple{kc, nc, std::clamp(even, S.mr, mc)};
            } else {
                return std::tuple{kc, nc, mc};
            }
        }();
        auto &Bpacked = packing_buffer<BPanels<T, S>>(
            round_up(std::min(nc_max, C.cols()), S.nr) / S.nr * kc_max, S.nr);

        for (auto [j, nc] : blocks(C.cols(), nc_max)) {
            const auto jslice = zipper::slice(j, nc);
            for (auto [p, kc] : blocks(A.cols(), kc_max)) {
                const auto kslice = zipper::slice(p, kc);
                // Packed once per (jc, pc), shared (read-only) by every mc
                // block of A.
                pack_B<T, S>(B.slice(kslice, jslice), Bpacked);

                // One row block of C: packs its own A (thread_local buffer)
                // and writes rows disjoint from every other block.
                const auto row_block = [&](index_type i) {
                    const index_type mc = std::min(mc_step, A.rows() - i);
                    const auto islice = zipper::slice(i, mc);
                    auto &Apacked = packing_buffer<APanels<T, S>>(
                        S.mr, round_up(mc_step, S.mr) / S.mr * kc_max);
                    pack_A<T, S>(A.slice(islice, kslice), Apacked, alpha);
                    auto Cblock = C.slice(islice, jslice);
                    if (p == 0) {
                        inner_kernel<T, S>(Apacked, Bpacked, kc, Cblock, beta);
                    } else {
                        inner_kernel<T, S>(Apacked,
                                           Bpacked,
                                           kc,
                                           Cblock,
                                           zipper::cw<1>);
                    }
                };
                const auto starts =
                    std::views::iota(index_type{0},
                                     (A.rows() + mc_step - 1) / mc_step)
                    | std::views::transform(
                        [&](index_type b) { return b * mc_step; });
                if constexpr (parallel) {
                    // Parallel algorithms want forward iterators over real
                    // references; materialize the (few) block offsets.
                    const auto offsets =
                        starts | std::ranges::to<std::vector>();
                    std::for_each(config.policy,
                                  offsets.begin(),
                                  offsets.end(),
                                  row_block);
                } else {
                    std::ranges::for_each(starts, row_block);
                }
            }
        }
    }
}

/// C = A * B (alpha = 1, beta = 0).
template <typename T,
          Tile S = default_tile<T>,
          ExecutionPolicy Policy = std::execution::sequenced_policy>
    requires SimdTile<S, simd_width<T>>
void gemm_blocked(zipper::concepts::Matrix auto const &A,
                  zipper::concepts::Matrix auto const &B,
                  zipper::concepts::Matrix auto &&C,
                  GemmConfig<T, S, Policy> const &config = {}) {
    gemm_blocked<T, S>(zipper::cw<1>,
                       A,
                       B,
                       zipper::cw<0>,
                       C,
                       config);
}

/// Entry point from MatrixProduct::assign_to / accumulate_to on raw
/// expressions: C = beta * C + alpha * A * B (precondition:
/// detail::GemmEligible).
///
/// Aliasing is AssignHelper's responsibility: `C = A * B` evaluates into a
/// fresh temporary, and `C.noalias() = A * B` is the caller's promise. The
/// kernel only asserts its preconditions.
template <typename Alpha,
          typename AExpr,
          typename BExpr,
          typename Beta,
          typename CExpr,
          ExecutionPolicy Policy = std::execution::sequenced_policy>
void gemm(Alpha alpha,
          const AExpr &A_,
          const BExpr &B_,
          Beta beta,
          CExpr &C_,
          const Policy &policy = {}) {
    using T = std::remove_cvref_t<
        typename zipper::expression::detail::ExpressionTraits<
            std::remove_cvref_t<CExpr>>::element_type>;
    const zipper::MatrixBase<const AExpr &> A(std::in_place, A_);
    const zipper::MatrixBase<const BExpr &> B(std::in_place, B_);
    zipper::MatrixBase<CExpr &> C(std::in_place, C_);

    ZIPPER_ASSERT(A.cols() == B.rows());
    ZIPPER_ASSERT(C.rows() == A.rows() && C.cols() == B.cols());
    // Cheap partial check of the no-alias contract: an operand sharing C's
    // buffer base (C itself, or a transpose of it).
    [[maybe_unused]] auto shares_buffer = [&](const auto &e) {
        if constexpr (requires { e.data(); C_.data(); }) {
            return static_cast<const void *>(e.data())
                   == static_cast<const void *>(C_.data());
        } else {
            return false;
        }
    };
    ZIPPER_ASSERT(!shares_buffer(A_) && !shares_buffer(B_));

    constexpr Tile S = default_tile<T>;
    gemm_blocked<T, S>(
        alpha, A, B, beta, C, GemmConfig<T, S, Policy>{.policy = policy});
}

} // namespace zipper::expression::binary::detail::gemm

#endif
