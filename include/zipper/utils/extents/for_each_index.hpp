#if !defined(ZIPPER_UTILS_EXTENTS_FOR_EACH_INDEX_HPP)
#define ZIPPER_UTILS_EXTENTS_FOR_EACH_INDEX_HPP
#include "zipper/concepts/Extents.hpp"
#include "zipper/detail/LayoutPreference.hpp"
#include "zipper/types.hpp"
#include <utility>

// GCC only fully unrolls short constant-trip loops early enough to keep small
// static objects (e.g. Vector<double, 4>) in registers at -O3; this hint gets
// the same result at -O2. Clang honours it too; other compilers ignore it.
#if defined(__GNUC__)
#define ZIPPER_UNROLL_SMALL _Pragma("GCC unroll 8")
#else
#define ZIPPER_UNROLL_SMALL
#endif

namespace zipper::utils::extents {

namespace detail {
    // Layout policy named by a layout policy or a layout preference type.
    template <typename L> struct layout_of {
        using type = L;
    };
    template <typename L>
        requires requires { typename L::layout_policy; }
    struct layout_of<L> {
        using type = typename L::layout_policy;
    };
    template <> struct layout_of<zipper::detail::NoLayoutPreference> {
        using type = zipper::storage::layout_right;
    };

    // Visit all index combinations; N dimensions remain to be looped over.
    // Row-major loops dims 0, 1, ... (outermost first) and appends each index;
    // column-major loops dims rank-1, rank-2, ... and prepends each index.
    // Either way fn receives indices in natural order (i0, i1, ...).
    template <bool ColMajor,
              rank_type N,
              zipper::concepts::Extents Extents,
              typename Fn,
              typename... Is>
    void visit(const Extents &ext, Fn &fn, Is... is) {
        if constexpr (N == 0) {
            fn(is...);
        } else {
            constexpr rank_type D = ColMajor ? N - 1 : Extents::rank() - N;
            ZIPPER_UNROLL_SMALL
            for (index_type i = 0; i < ext.extent(D); ++i) {
                if constexpr (ColMajor) {
                    visit<ColMajor, N - 1>(ext, fn, i, is...);
                } else {
                    visit<ColMajor, N - 1>(ext, fn, is..., i);
                }
            }
        }
    }
} // namespace detail

/// Iterate over all index combinations of `ext` in the order given by
/// `Layout`, calling fn(i0, i1, ..., i_{rank-1}) (always natural order).
/// `Layout` is a layout policy (layout_left -> column-major, layout_right ->
/// row-major) or a layout preference (DenseLayoutPreference<L> uses L;
/// NoLayoutPreference -> row-major).
template <typename Layout, zipper::concepts::Extents Extents, typename Fn>
void for_each_index(const Extents &ext, Fn &&fn) {
    using layout = typename detail::layout_of<Layout>::type;
    constexpr bool col_major =
        std::is_same_v<layout, zipper::storage::layout_left>;
    detail::visit<col_major, Extents::rank()>(ext, fn);
}

} // namespace zipper::utils::extents
#endif
