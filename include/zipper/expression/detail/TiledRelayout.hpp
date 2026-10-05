#if !defined(ZIPPER_EXPRESSION_DETAIL_TILEDRELAYOUT_HPP)
#define ZIPPER_EXPRESSION_DETAIL_TILEDRELAYOUT_HPP

/// @file TiledRelayout.hpp
/// @brief Cache-blocked assignment for relayouts: transposes, row-/column-major
/// changes, tensor permutations.
///
/// When the source and destination of an assignment are linear arrays whose
/// fastest dimensions differ, walking either one in order strides the other
/// across the whole array; once the operands outgrow the cache nearly every
/// access misses. Copying those two dimensions in square tiles keeps both
/// sides within a few cache lines.
///
/// Whether an assignment is a relayout is decided from the types — the
/// layouts of the two mapping types (storage::fastest_dimension_v); only the
/// size of dynamic extents is checked at run time. Every other assignment keeps the coefficient path, which the
/// compiler already turns into memmove-speed or vectorized loops.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "zipper/expression/concepts/capabilities.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/storage/layout_traits.hpp"
#include "zipper/types.hpp"

namespace zipper::expression::detail {

namespace tiled_relayout_detail {
    /// Tile edge, in elements.
    inline constexpr index_type tile = 32;

    /// Below this many bytes (source + destination) the operands fit in
    /// cache and the coefficient path is faster.
#if defined(ZIPPER_RELAYOUT_TILING_MIN_BYTES)
    inline constexpr std::size_t min_bytes = ZIPPER_RELAYOUT_TILING_MIN_BYTES;
#else
    inline constexpr std::size_t min_bytes = std::size_t(256) * 1024;
#endif

    template <typename E>
    using traits = ExpressionTraits<std::remove_cvref_t<E>>;

    /// Fastest dimension of E's mapping() type.
    template <typename E>
    inline constexpr rank_type fastest = storage::fastest_dimension_v<
        std::remove_cvref_t<decltype(std::declval<const E &>().mapping())>>;

    template <typename From, typename To>
    inline constexpr std::size_t element_bytes =
        sizeof(std::remove_cvref_t<decltype(*std::declval<From &>().data())>)
        + sizeof(std::remove_cvref_t<decltype(*std::declval<To &>().data())>);

    /// Bytes moved, when the shape is fully static.
    template <typename From, typename To>
    constexpr auto static_bytes() -> std::size_t {
        using E = typename traits<To>::extents_type;
        std::size_t n = 1;
        for (rank_type d = 0; d < E::rank(); ++d) { n *= E::static_extent(d); }
        return n * element_bytes<From, To>;
    }
} // namespace tiled_relayout_detail

/// `to = from` is a relayout worth tiling as far as the types can tell:
/// both are linear arrays of the same rank whose layouts have known,
/// different fastest dimensions, the destination buffer is writable, and — for fully static
/// shapes — the operands are large enough.
template <typename From, typename To>
concept TiledRelayout = [] {
    namespace tr = tiled_relayout_detail;
    using F = tr::traits<From>;
    using T = tr::traits<To>;
    if constexpr (!concepts::LinearArray<From> || !concepts::LinearArray<To>) {
        return false;
    } else if constexpr (F::extents_type::rank() != T::extents_type::rank()
                         || T::extents_type::rank() < 2) {
        return false;
    } else if constexpr (tr::fastest<From> == storage::unknown_dimension
                         || tr::fastest<To> == storage::unknown_dimension
                         || tr::fastest<From> == tr::fastest<To>) {
        return false;
    } else if constexpr (!requires(To &to, const From &from) {
                             *to.data() = *from.data();
                         }) {
        return false;
    } else if constexpr (T::extents_type::rank_dynamic() == 0) {
        return tr::static_bytes<From, To>() >= tr::min_bytes;
    } else {
        return true;
    }
}();

namespace tiled_relayout_detail {
    /// dst[k * ds] = src[k * ss] for k in [0, n). The ranges never overlap
    /// (AssignHelper has ruled out aliasing), hence __restrict; one loop per
    /// stride pattern so the contiguous side vectorizes.
    template <typename D, typename S>
    [[gnu::always_inline]] inline void run(D *__restrict dst,
                                           index_type ds,
                                           const S *__restrict src,
                                           index_type ss,
                                           index_type n) {
        if (ds == 1 && ss == 1) {
            if constexpr (std::is_same_v<std::remove_cv_t<D>, std::remove_cv_t<S>>
                          && std::is_trivially_copyable_v<D>) {
                std::memcpy(dst, src, n * sizeof(D));
            } else {
                for (index_type k = 0; k < n; ++k) { dst[k] = src[k]; }
            }
        } else if (ds == 1) {
            for (index_type k = 0; k < n; ++k) { dst[k] = src[k * ss]; }
        } else if (ss == 1) {
            for (index_type k = 0; k < n; ++k) { dst[k * ds] = src[k]; }
        } else {
            for (index_type k = 0; k < n; ++k) { dst[k * ds] = src[k * ss]; }
        }
    }

    /// The dimensions other than `a` and `b`, outermost first.
    template <std::size_t R, rank_type a, rank_type b>
    constexpr auto outer_dims() -> std::array<rank_type, R - 2> {
        std::array<rank_type, R - 2> out{};
        std::size_t k = 0;
        for (rank_type d = 0; d < R; ++d) {
            if (d != a && d != b) { out[k++] = d; }
        }
        return out;
    }
} // namespace tiled_relayout_detail

/// Whether a TiledRelayout is large enough to tile: a constant for fully
/// static shapes (the concept already checked it), the dynamic size otherwise.
template <typename From, typename To>
    requires TiledRelayout<From, To>
constexpr auto worth_tiling(const To &to) -> bool {
    namespace tr = tiled_relayout_detail;
    using E = typename tr::traits<To>::extents_type;
    if constexpr (E::rank_dynamic() == 0) {
        return true;
    } else {
        std::size_t n = 1;
        for (rank_type d = 0; d < E::rank(); ++d) { n *= to.extent(d); }
        return n * tr::element_bytes<From, To> >= tr::min_bytes;
    }
}

namespace tiled_relayout_detail {
    /// A relayout reduced to plain data: buffers, extents, strides.
    template <typename D, typename S, std::size_t R>
    struct Plan {
        D *dst;
        const S *src;
        std::array<index_type, R> extent;
        std::array<index_type, R> ds; // destination strides
        std::array<index_type, R> ss; // source strides
    };

    /// The copy itself, on plain data passed by value and kept out of line.
    /// Taking the expressions by reference here instead would let their
    /// addresses escape from the (inlined) assignment, and the compiler then
    /// generates worse code for the small-size coefficient path next to it.
    template <rank_type inner, rank_type src_inner, typename D, typename S,
              std::size_t R>
    [[gnu::noinline]] void copy(Plan<D, S, R> p) {
        constexpr auto outer = outer_dims<R, inner, src_inner>();
        const index_type n_in = p.extent[inner], n_s = p.extent[src_inner];
        const index_type ds = p.ds[inner], ss = p.ss[inner];
        const index_type dso = p.ds[src_inner], sso = p.ss[src_inner];

        const auto copy_planes = [&](index_type doff, index_type soff) {
            for (index_type o0 = 0; o0 < n_s; o0 += tile) {
                const index_type o1 = std::min(n_s, o0 + tile);
                for (index_type i0 = 0; i0 < n_in; i0 += tile) {
                    const index_type i1 = std::min(n_in, i0 + tile);
                    for (index_type o = o0; o < o1; ++o) {
                        run(p.dst + doff + o * dso + i0 * ds,
                            ds,
                            p.src + soff + o * sso + i0 * ss,
                            ss,
                            i1 - i0);
                    }
                }
            }
        };

        if constexpr (R == 2) {
            copy_planes(0, 0);
        } else {
            // Odometer over the outer dimensions (last one fastest), offsets
            // kept incrementally.
            for (rank_type k = 0; k < R - 2; ++k) {
                if (p.extent[outer[k]] == 0) { return; }
            }
            std::array<index_type, R - 2> idx{};
            index_type doff = 0, soff = 0;
            while (true) {
                copy_planes(doff, soff);
                std::size_t k = R - 2;
                while (k-- > 0) {
                    const rank_type d = outer[k];
                    if (++idx[k] < p.extent[d]) {
                        doff += p.ds[d];
                        soff += p.ss[d];
                        break;
                    }
                    doff -= (p.extent[d] - 1) * p.ds[d];
                    soff -= (p.extent[d] - 1) * p.ss[d];
                    idx[k] = 0;
                    if (k == 0) { return; }
                }
            }
        }
    }
} // namespace tiled_relayout_detail

/// to = from for a TiledRelayout (shapes already match): tiles over the two
/// fastest dimensions, runs along the destination's inside each tile, and
/// an outer loop over the remaining dimensions.
template <typename From, typename To>
    requires TiledRelayout<From, To>
void tiled_relayout(const From &from, To &to) {
    namespace tr = tiled_relayout_detail;
    constexpr std::size_t R = tr::traits<To>::extents_type::rank();
    const auto tm = to.mapping();
    const auto fm = from.mapping();
    tr::Plan<std::remove_reference_t<decltype(*to.data())>,
             std::remove_cvref_t<decltype(*from.data())>,
             R>
        p{to.data(), from.data(), {}, {}, {}};
    for (rank_type d = 0; d < R; ++d) {
        p.extent[d] = to.extent(d);
        p.ds[d] = tm.stride(d);
        p.ss[d] = fm.stride(d);
    }
    tr::copy<tr::fastest<To>, tr::fastest<From>>(p);
}

} // namespace zipper::expression::detail

#endif
