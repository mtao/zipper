#pragma once

/// @file IndexSet.hpp
/// @brief Lightweight range types for describing non-zero index patterns.
/// @ingroup sparsity
///
/// Expressions with known structural zeros (TriangularView, Identity, Unit,
/// etc.) report their non-zero index patterns via `index_set<D>(...)`.
/// The return type is expression-specific — each expression returns the most
/// efficient range type for its sparsity structure.
///
/// All range types satisfy the `IndexSet` concept and provide:
///   - `contains(idx)` — O(1) membership test (O(log n) for SparseIndexRange)
///   - `begin()` / `end()` — forward iteration over non-zero indices
///   - `size()` — number of non-zero indices
///   - `empty()` — whether the range is empty
///
/// Range types:
///   - `ContiguousIndexRange` — half-open interval [first, last)
///   - `SingleIndexRange`     — exactly one index
///   - `SparseIndexRange`     — sorted set of indices
///   - `FullRange`            — all indices in [0, extent)
///
/// @see zipper::expression::unary::TriangularView — returns
///      ContiguousIndexRange from index_set (triangular regions).
/// @see zipper::expression::nullary::Identity — returns SingleIndexRange
///      from index_set (one non-zero per row/column).
/// @see zipper::expression::nullary::Unit — returns SingleIndexRange
///      from index_set (one non-zero in a rank-1 expression).
/// @see zipper::expression::binary::ZeroAwareOperation — propagates
///      index_set through addition/subtraction (union semantics).
/// @see zipper::expression::binary::MatrixProduct — uses index_set
///      for intersection semantics in dot product loops.
/// @see zipper::expression::binary::MatrixVectorProduct — uses index_set
///      for intersection semantics in matrix-vector dot product loops.

#include <algorithm>
#include <array>
#include <ranges>
#include <span>
#include <vector>

#include "zipper/detail/no_unique_address.hpp"
#include "zipper/types.hpp"

namespace zipper::expression::detail {

// ═════════════════════════════════════════════════════════════════════════════
// Templated index set types
//
// Each field can be either `index_type` (runtime) or
// `std::integral_constant<index_type, N>` (compile-time).  With
// `ZIPPER_NO_UNIQUE_ADDRESS`, compile-time fields occupy zero storage.
// CTAD guides map integral arguments to `index_type` and preserve
// `integral_constant` types, mirroring `strided_slice`.
// ═════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
/// @brief An evenly-strided half-open index set using range(first, last, stride)
///        semantics.
///
/// Represents indices {first, first+stride, first+2*stride, ...} where each
/// index is strictly less than `last`.  Fields use `ZIPPER_NO_UNIQUE_ADDRESS` so
/// compile-time `std::integral_constant` parameters occupy zero storage.
///
/// When StrideType is `std::integral_constant<index_type, 1>` the set is
/// contiguous: [first, last).  The `contains()` hot path compiles to a simple
/// range check with no modulo in this case.
///
/// ContiguousIndexSet<FT, LT> is a type alias for
/// StridedIndexSet<FT, LT, static_index_t<1>>, so all existing code that
/// uses `.first` / `.last` on contiguous ranges continues to work unchanged.
///
/// @tparam FirstType   Type of the inclusive lower bound.
/// @tparam LastType    Type of the exclusive upper bound.
/// @tparam StrideType  Type of the step size.
// ─────────────────────────────────────────────────────────────────────────────
template <typename FirstType = index_type, typename LastType = index_type,
          typename StrideType = index_type>
struct StridedIndexSet {
    ZIPPER_NO_UNIQUE_ADDRESS FirstType first;   ///< Inclusive lower bound.
    ZIPPER_NO_UNIQUE_ADDRESS LastType last;     ///< Exclusive upper bound.
    ZIPPER_NO_UNIQUE_ADDRESS StrideType stride; ///< Step between consecutive elements.

    /// @brief Test whether @p idx belongs to this strided set.
    ///
    /// When stride is compile-time 1, this is a contiguous range check.
    /// Otherwise: check in-bounds and divisibility by stride.
    constexpr auto contains(index_type idx) const -> bool {
        const auto f = index_type(first);
        const auto l = index_type(last);
        const auto s = index_type(stride);
        if (f >= l || idx < f || idx >= l) return false;

        if constexpr (requires { StrideType::value; }) {
            if constexpr (StrideType::value == 1) {
                return true;  // already checked f <= idx < l
            } else {
                return (idx - f) % s == 0;
            }
        } else {
            if (s == 1) return true;
            return (idx - f) % s == 0;
        }
    }

    constexpr auto empty() const -> bool {
        return index_type(first) >= index_type(last);
    }

    /// @brief Number of indices: ceil((last - first) / stride).
    constexpr auto size() const -> index_type {
        if (empty()) return index_type{0};
        const auto span = index_type(last) - index_type(first);
        const auto s = index_type(stride);
        return (span + s - 1) / s;
    }

    /// @brief Forward iterator that generates the strided sequence.
    struct Iterator {
        index_type current;
        index_type sentinel;
        index_type step;

        using value_type = index_type;
        using difference_type = std::ptrdiff_t;

        constexpr auto operator*() const -> index_type { return current; }
        constexpr auto operator++() -> Iterator & {
            current += step;
            return *this;
        }
        constexpr auto operator++(int) -> Iterator {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
        constexpr auto operator==(const Iterator &other) const -> bool {
            // Both at-or-past-end → equal (handles stride overshoot).
            const bool a_end = current >= sentinel;
            const bool b_end = other.current >= other.sentinel;
            if (a_end && b_end) return true;
            if (a_end != b_end) return false;
            return current == other.current;
        }
        constexpr auto operator!=(const Iterator &other) const -> bool {
            return !(*this == other);
        }
    };

    constexpr auto begin() const -> Iterator {
        return {index_type(first), index_type(last), index_type(stride)};
    }
    constexpr auto end() const -> Iterator {
        return {index_type(last), index_type(last), index_type(stride)};
    }
};

/// CTAD: integral types → index_type, integral_constant → preserved.
template <typename FT, typename LT, typename ST>
StridedIndexSet(FT, LT, ST) -> StridedIndexSet<
    std::conditional_t<std::is_integral_v<std::decay_t<FT>>,
                       index_type, std::decay_t<FT>>,
    std::conditional_t<std::is_integral_v<std::decay_t<LT>>,
                       index_type, std::decay_t<LT>>,
    std::conditional_t<std::is_integral_v<std::decay_t<ST>>,
                       index_type, std::decay_t<ST>>>;

/// Two-argument CTAD: deduces contiguous (stride = static_index_t<1>).
template <typename FT, typename LT>
StridedIndexSet(FT, LT) -> StridedIndexSet<
    std::conditional_t<std::is_integral_v<std::decay_t<FT>>,
                       index_type, std::decay_t<FT>>,
    std::conditional_t<std::is_integral_v<std::decay_t<LT>>,
                       index_type, std::decay_t<LT>>,
    static_index_t<1>>;

/// Contiguous index set: StridedIndexSet with stride fixed at 1.
template <typename FirstType = index_type, typename LastType = index_type>
using ContiguousIndexSet = StridedIndexSet<FirstType, LastType, static_index_t<1>>;

/// Backward-compatible alias: fully-runtime contiguous index range.
using ContiguousIndexRange = ContiguousIndexSet<>;

/// Backward-compatible alias: fully-runtime strided index range.
using StridedIndexRange = StridedIndexSet<>;

// ─────────────────────────────────────────────────────────────────────────────
/// @brief A range containing exactly one index.
///
/// Used by Identity (rank-2) and Diagonal expressions, where each row has
/// exactly one non-zero column.
// ─────────────────────────────────────────────────────────────────────────────
template <typename ValueType = index_type>
struct SingleIndexSet {
    ZIPPER_NO_UNIQUE_ADDRESS ValueType value;  ///< The single non-zero index.

    /// @brief Test whether @p idx equals the stored value.
    constexpr auto contains(index_type idx) const -> bool {
        return idx == index_type(value);
    }

    constexpr auto empty() const -> bool { return false; }
    constexpr auto size() const -> index_type { return 1; }

    auto begin() const {
        return std::ranges::iota_view(index_type(value),
                                       index_type(value) + 1)
            .begin();
    }
    auto end() const {
        return std::ranges::iota_view(index_type(value),
                                       index_type(value) + 1)
            .end();
    }
};

/// CTAD for SingleIndexSet.
template <typename VT>
SingleIndexSet(VT) -> SingleIndexSet<
    std::conditional_t<std::is_integral_v<std::decay_t<VT>>,
                       index_type, std::decay_t<VT>>>;

/// Backward-compatible alias: fully-runtime single index range.
using SingleIndexRange = SingleIndexSet<>;

// ─────────────────────────────────────────────────────────────────────────────
/// @brief An index set that is always empty — contains no indices.
///
/// Used by expressions that are structurally zero everywhere (e.g.
/// StaticConstant<T, 0, ...>).  Satisfies IndexSet with zero storage.
// ─────────────────────────────────────────────────────────────────────────────
struct EmptyIndexRange {
    constexpr auto contains([[maybe_unused]] index_type idx) const -> bool {
        return false;
    }

    constexpr auto empty() const -> bool { return true; }
    constexpr auto size() const -> index_type { return 0; }

    /// begin() == end() (empty range).
    auto begin() const {
        return std::ranges::iota_view(index_type{0}, index_type{0}).begin();
    }
    auto end() const {
        return std::ranges::iota_view(index_type{0}, index_type{0}).end();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Sparse index set types
//
// Three ownership levels for sorted sparse index sets, mirroring the
// expression_storage_t convention (lvalue → non-owning, rvalue → owning):
//
//   StaticSparseIndexSet<N>  — std::array<index_type, N>, compile-time size
//   SpanSparseIndexSet       — std::span<const index_type>, non-owning
//   DynamicSparseIndexSet    — std::vector<index_type>, owning
//
// All require indices sorted ascending.  contains() uses binary search.
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Sparse index set with compile-time-known size, backed by std::array.
///
/// Useful when the number of non-zero indices is known at compile time
/// (e.g., the two off-diagonal entries in a tridiagonal row).
template <std::size_t N>
struct StaticSparseIndexSet {
    std::array<index_type, N> indices;  ///< Sorted ascending.

    constexpr auto contains(index_type idx) const -> bool {
        // For small N, linear scan beats binary search.
        if constexpr (N <= 8) {
            for (std::size_t i = 0; i < N; ++i) {
                if (indices[i] == idx) return true;
                if (indices[i] > idx) return false;  // sorted, early exit
            }
            return false;
        } else {
            return std::binary_search(indices.begin(), indices.end(), idx);
        }
    }

    constexpr auto empty() const -> bool { return N == 0; }
    constexpr auto size() const -> index_type { return N; }

    constexpr auto begin() const { return indices.begin(); }
    constexpr auto end() const { return indices.end(); }
};

/// @brief Non-owning sparse index set backed by std::span.
///
/// Borrows a sorted index buffer from the caller.  Use when the lifetime
/// of the underlying storage is guaranteed (e.g., referencing a static
/// array or a vector that outlives the expression tree).
///
/// Call `to_owned()` to produce a DynamicSparseIndexSet that owns its data.
struct SpanSparseIndexSet {
    std::span<const index_type> indices;  ///< Sorted ascending, non-owning.

    auto contains(index_type idx) const -> bool {
        return std::binary_search(indices.begin(), indices.end(), idx);
    }

    auto empty() const -> bool { return indices.empty(); }
    auto size() const -> index_type { return indices.size(); }

    auto begin() const { return indices.begin(); }
    auto end() const { return indices.end(); }

    /// @brief Create an owning copy of this span's data.
    inline auto to_owned() const -> struct DynamicSparseIndexSet;
};

/// @brief Owning sparse index set backed by std::vector.
///
/// The default choice for runtime-constructed sparse index sets.
/// This is the type that the old `SparseIndexRange` was.
struct DynamicSparseIndexSet {
    std::vector<index_type> indices;  ///< Sorted ascending.

    auto contains(index_type idx) const -> bool {
        return std::binary_search(indices.begin(), indices.end(), idx);
    }

    auto empty() const -> bool { return indices.empty(); }
    auto size() const -> index_type { return indices.size(); }

    auto begin() const { return indices.begin(); }
    auto end() const { return indices.end(); }

    /// @brief Create a non-owning view of this set's data.
    auto as_span() const -> SpanSparseIndexSet {
        return {std::span<const index_type>(indices)};
    }
};

/// Deferred definition (DynamicSparseIndexSet must be complete).
inline auto SpanSparseIndexSet::to_owned() const -> DynamicSparseIndexSet {
    return {std::vector<index_type>(indices.begin(), indices.end())};
}

/// Backward-compatible alias.
using SparseIndexRange = DynamicSparseIndexSet;

// ─────────────────────────────────────────────────────────────────────────────
/// @brief The full index range [0, extent) — every position is non-zero.
///
/// Used as the default for dense expressions, enabling universal code paths
/// without `if constexpr` branching when desired.
// ─────────────────────────────────────────────────────────────────────────────
template <typename ExtentType = index_type>
struct FullIndexSet {
    ZIPPER_NO_UNIQUE_ADDRESS ExtentType extent;  ///< Range is [0, extent).

    /// @brief Always returns true (no known zeros).
    constexpr auto contains([[maybe_unused]] index_type idx) const -> bool {
        return true;
    }

    constexpr auto empty() const -> bool { return index_type(extent) == 0; }
    constexpr auto size() const -> index_type { return index_type(extent); }

    auto begin() const {
        return std::ranges::iota_view(index_type{0}, index_type(extent))
            .begin();
    }
    auto end() const {
        return std::ranges::iota_view(index_type{0}, index_type(extent))
            .end();
    }
};

/// CTAD for FullIndexSet.
template <typename ET>
FullIndexSet(ET) -> FullIndexSet<
    std::conditional_t<std::is_integral_v<std::decay_t<ET>>,
                       index_type, std::decay_t<ET>>>;

/// Backward-compatible alias: fully-runtime full range.
using FullRange = FullIndexSet<>;

// ═════════════════════════════════════════════════════════════════════════════
// Detection traits for templated index set types
// ═════════════════════════════════════════════════════════════════════════════

/// @brief Trait to detect EmptyIndexRange.
template <typename T>
struct is_empty_index_range : std::false_type {};
template <>
struct is_empty_index_range<EmptyIndexRange> : std::true_type {};
template <typename T>
concept IsEmptyIndexRange =
    is_empty_index_range<std::remove_cvref_t<T>>::value;

/// @brief Trait to detect any ContiguousIndexSet specialization (stride = static_index_t<1>).
template <typename T>
struct is_contiguous_index_set : std::false_type {};
template <typename FT, typename LT>
struct is_contiguous_index_set<StridedIndexSet<FT, LT, static_index_t<1>>>
    : std::true_type {};
template <typename T>
concept IsContiguousIndexSet =
    is_contiguous_index_set<std::remove_cvref_t<T>>::value;

/// @brief Trait to detect any SingleIndexSet specialization.
template <typename T>
struct is_single_index_set : std::false_type {};
template <typename VT>
struct is_single_index_set<SingleIndexSet<VT>> : std::true_type {};
template <typename T>
concept IsSingleIndexSet =
    is_single_index_set<std::remove_cvref_t<T>>::value;

/// @brief Trait to detect any FullIndexSet specialization.
template <typename T>
struct is_full_index_set : std::false_type {};
template <typename ET>
struct is_full_index_set<FullIndexSet<ET>> : std::true_type {};
template <typename T>
concept IsFullIndexSet =
    is_full_index_set<std::remove_cvref_t<T>>::value;

/// @brief Trait to detect any StridedIndexSet specialization.
template <typename T>
struct is_strided_index_set : std::false_type {};
template <typename OT, typename ET, typename ST>
struct is_strided_index_set<StridedIndexSet<OT, ET, ST>> : std::true_type {};
template <typename T>
concept IsStridedIndexSet =
    is_strided_index_set<std::remove_cvref_t<T>>::value;

/// @brief Trait to detect StaticSparseIndexSet<N>.
template <typename T>
struct is_static_sparse_index_set : std::false_type {};
template <std::size_t N>
struct is_static_sparse_index_set<StaticSparseIndexSet<N>> : std::true_type {};
template <typename T>
concept IsStaticSparseIndexSet =
    is_static_sparse_index_set<std::remove_cvref_t<T>>::value;

/// @brief Trait to detect SpanSparseIndexSet.
template <typename T>
struct is_span_sparse_index_set : std::false_type {};
template <>
struct is_span_sparse_index_set<SpanSparseIndexSet> : std::true_type {};
template <typename T>
concept IsSpanSparseIndexSet =
    is_span_sparse_index_set<std::remove_cvref_t<T>>::value;

/// @brief Trait to detect DynamicSparseIndexSet.
template <typename T>
struct is_dynamic_sparse_index_set : std::false_type {};
template <>
struct is_dynamic_sparse_index_set<DynamicSparseIndexSet> : std::true_type {};
template <typename T>
concept IsDynamicSparseIndexSet =
    is_dynamic_sparse_index_set<std::remove_cvref_t<T>>::value;

/// @brief Concept matching any sparse index set type.
template <typename T>
concept IsSparseIndexSet =
    IsStaticSparseIndexSet<T> || IsSpanSparseIndexSet<T> ||
    IsDynamicSparseIndexSet<T>;

// ─────────────────────────────────────────────────────────────────────────────
/// @brief Concept satisfied by all index set types.
///
/// Any type that provides `contains(idx)`, forward iteration via
/// `begin()`/`end()`, and `size()`/`empty()` satisfies this concept.
// ─────────────────────────────────────────────────────────────────────────────
template <typename R>
concept IndexSet = requires(const R &r, index_type idx) {
    { r.contains(idx) } -> std::convertible_to<bool>;
    { r.begin() };
    { r.end() };
    { r.size() } -> std::convertible_to<index_type>;
    { r.empty() } -> std::convertible_to<bool>;
};

/// Backward-compatible alias for the old concept name.
template <typename R>
concept NonzeroRange = IndexSet<R>;

// Static assertions to verify all range types satisfy the concept.
static_assert(IndexSet<EmptyIndexRange>);
static_assert(IndexSet<ContiguousIndexRange>);
static_assert(IndexSet<SingleIndexRange>);
static_assert(IndexSet<SparseIndexRange>);
static_assert(IndexSet<FullRange>);
static_assert(IndexSet<StridedIndexRange>);
static_assert(IndexSet<StaticSparseIndexSet<3>>);
static_assert(IndexSet<SpanSparseIndexSet>);
static_assert(IndexSet<DynamicSparseIndexSet>);

// Also verify non-default specializations.
static_assert(IndexSet<ContiguousIndexSet<
    std::integral_constant<index_type, 0>,
    std::integral_constant<index_type, 5>>>);
static_assert(IndexSet<SingleIndexSet<
    std::integral_constant<index_type, 3>>>);
static_assert(IndexSet<FullIndexSet<
    std::integral_constant<index_type, 10>>>);
static_assert(IndexSet<StridedIndexSet<
    std::integral_constant<index_type, 0>,
    std::integral_constant<index_type, 12>,
    std::integral_constant<index_type, 3>>>);

// ─────────────────────────────────────────────────────────────────────────────
// Forward declaration for DisjointRange (needed by concept trait below).
// ─────────────────────────────────────────────────────────────────────────────
template <IndexSet... Ranges>
struct DisjointRange;

/// @brief Trait to detect DisjointRange types.
template <typename T>
struct is_disjoint_range : std::false_type {};

template <IndexSet... Rs>
struct is_disjoint_range<DisjointRange<Rs...>> : std::true_type {};

template <typename T>
concept IsDisjointRange = is_disjoint_range<std::remove_cvref_t<T>>::value;

// ─────────────────────────────────────────────────────────────────────────────
/// @brief Convert any IndexSet to a ContiguousIndexRange (bounding box).
///
/// Used by range_union and other utilities that need a uniform type.
///   - ContiguousIndexRange → identity
///   - SingleIndexRange     → [value, value+1)
///   - FullRange            → [0, extent)
///   - SparseIndexRange     → bounding interval
// ─────────────────────────────────────────────────────────────────────────────
constexpr auto to_contiguous_range(const ContiguousIndexRange &r)
    -> ContiguousIndexRange {
    return r;
}

constexpr auto to_contiguous_range([[maybe_unused]] const EmptyIndexRange &)
    -> ContiguousIndexRange {
    return {index_type{0}, index_type{0}};
}

constexpr auto to_contiguous_range(const SingleIndexRange &r)
    -> ContiguousIndexRange {
    return {r.value, r.value + 1};
}

constexpr auto to_contiguous_range(const FullRange &r)
    -> ContiguousIndexRange {
    return {index_type{0}, r.extent};
}

/// @brief Bounding box of a StridedIndexSet: [first, first + (size-1)*stride + 1).
///
/// For contiguous sets (stride=1), this is identity.
/// For general stride, returns the tightest contiguous range covering all
/// elements.
template <typename FT, typename LT, typename ST>
    requires(!IsContiguousIndexSet<StridedIndexSet<FT, LT, ST>>)
constexpr auto to_contiguous_range(const StridedIndexSet<FT, LT, ST> &r)
    -> ContiguousIndexRange {
    if (r.empty()) return {index_type{0}, index_type{0}};
    const auto f = index_type(r.first);
    const auto s = index_type(r.stride);
    const auto n = r.size();
    return {f, f + (n - 1) * s + 1};
}

/// @brief Bounding box of any sparse index set: [min, max+1).
///
/// Works for StaticSparseIndexSet, SpanSparseIndexSet, and
/// DynamicSparseIndexSet.  Since indices are sorted, first/last elements
/// give the bounds directly.
template <IsSparseIndexSet S>
inline auto to_contiguous_range(const S &r)
    -> ContiguousIndexRange {
    if (r.empty()) return {index_type{0}, index_type{0}};
    // Sorted ascending: first element is min, last element is max.
    const auto lo = static_cast<index_type>(*r.begin());
    const auto hi = static_cast<index_type>(*std::prev(r.end()));
    return {lo, hi + 1};
}

/// @brief Bounding box of a DisjointRange — the tightest ContiguousIndexRange
///        covering all segments.
template <IndexSet... Rs>
constexpr auto to_contiguous_range(const DisjointRange<Rs...> &dr)
    -> ContiguousIndexRange {
    if (dr.empty()) return {index_type{0}, index_type{0}};
    // Each segment is converted to its contiguous bounding box; we take
    // the overall [min(first), max(last)).
    index_type lo = std::dynamic_extent;
    index_type hi = 0;
    auto update = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (([&] {
            auto cr = to_contiguous_range(std::get<Is>(dr.segments));
            if (!cr.empty()) {
                lo = std::min(lo, cr.first);
                hi = std::max(hi, cr.last);
            }
        }()), ...);
    };
    update(std::index_sequence_for<Rs...>{});
    if (lo == std::dynamic_extent) return {index_type{0}, index_type{0}};
    return {lo, hi};
}

// ─────────────────────────────────────────────────────────────────────────────
/// @brief Chaining iterator for DisjointRange.
///
/// Walks through each sub-range's elements in order, skipping empty segments.
/// Provides forward iterator semantics suitable for range-for loops.
// ─────────────────────────────────────────────────────────────────────────────
template <IndexSet... Ranges>
class DisjointRangeIterator {
    static constexpr std::size_t N = sizeof...(Ranges);

    using ranges_tuple = std::tuple<Ranges...>;
    const ranges_tuple *m_ranges;

    // Store current position as variant of segment iterators.
    // We track which segment we're in and the current iterator position
    // within that segment.  For simplicity and performance with small N,
    // we store all begin/end iterators and the active segment index.
    //
    // However, since the iterator types may differ per segment, we use
    // a simpler approach: store the current linear index and segment index.
    index_type m_value;       ///< Current index value being pointed to.
    std::size_t m_segment;    ///< Which segment we're currently in.

    // Precomputed segment boundaries for fast advance.
    // segments_first[i] / segments_last[i] give the first/last of
    // the i-th segment (as ContiguousIndexRange bounds for iota-backed
    // ranges).  For truly generic ranges, we fall back to iterating.
    //
    // Since all practical segments are ContiguousIndexRange, we store
    // their bounds directly.
    std::array<index_type, N> m_seg_first;
    std::array<index_type, N> m_seg_last;

   public:
    using value_type = index_type;
    using difference_type = std::ptrdiff_t;

    /// @brief Construct a begin iterator.
    constexpr DisjointRangeIterator(const ranges_tuple &ranges,
                                     bool at_end)
        : m_ranges(&ranges), m_value{0}, m_segment{0} {
        // Extract bounds from each segment.
        extract_bounds(std::index_sequence_for<Ranges...>{});

        if (at_end) {
            m_segment = N;
            m_value = 0;
        } else {
            // Advance to first non-empty segment.
            m_segment = 0;
            skip_empty_segments();
            if (m_segment < N) {
                m_value = m_seg_first[m_segment];
            }
        }
    }

    constexpr auto operator*() const -> index_type { return m_value; }

    constexpr auto operator++() -> DisjointRangeIterator & {
        ++m_value;
        if (m_value >= m_seg_last[m_segment]) {
            ++m_segment;
            skip_empty_segments();
            if (m_segment < N) {
                m_value = m_seg_first[m_segment];
            }
        }
        return *this;
    }

    constexpr auto operator++(int) -> DisjointRangeIterator {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    constexpr auto operator==(const DisjointRangeIterator &other) const
        -> bool {
        if (m_segment == N && other.m_segment == N) return true;
        return m_segment == other.m_segment && m_value == other.m_value;
    }

    constexpr auto operator!=(const DisjointRangeIterator &other) const
        -> bool {
        return !(*this == other);
    }

   private:
    template <std::size_t... Is>
    constexpr void extract_bounds(std::index_sequence<Is...>) {
        ((extract_one_bound<Is>()), ...);
    }

    template <std::size_t I>
    constexpr void extract_one_bound() {
        auto cr = to_contiguous_range(std::get<I>(*m_ranges));
        m_seg_first[I] = cr.first;
        m_seg_last[I] = cr.last;
    }

    constexpr void skip_empty_segments() {
        while (m_segment < N &&
               m_seg_first[m_segment] >= m_seg_last[m_segment]) {
            ++m_segment;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
/// @brief A disjoint union of multiple non-zero index ranges.
///
/// Holds a variadic tuple of sub-ranges.  Segments must be ordered
/// (all indices in segment i < all indices in segment i+1) and
/// non-overlapping.  Empty segments are permitted and are skipped
/// during iteration.
///
/// Provides both the standard IndexSet interface (begin/end/contains/
/// size/empty) and a compile-time-unrolled `for_each(fn)` for zero-overhead
/// iteration in hot paths.
// ─────────────────────────────────────────────────────────────────────────────
template <IndexSet... Ranges>
struct DisjointRange {
    static constexpr std::size_t num_segments = sizeof...(Ranges);

    std::tuple<Ranges...> segments;

    /// @brief Test whether @p idx is in any segment.
    constexpr auto contains(index_type idx) const -> bool {
        return contains_impl(idx, std::index_sequence_for<Ranges...>{});
    }

    /// @brief Total number of indices across all segments.
    constexpr auto size() const -> index_type {
        return size_impl(std::index_sequence_for<Ranges...>{});
    }

    /// @brief True if all segments are empty.
    constexpr auto empty() const -> bool {
        return empty_impl(std::index_sequence_for<Ranges...>{});
    }

    auto begin() const {
        return DisjointRangeIterator<Ranges...>(segments, false);
    }

    auto end() const {
        return DisjointRangeIterator<Ranges...>(segments, true);
    }

    /// @brief Compile-time-unrolled iteration over all segments.
    ///
    /// Calls `fn(idx)` for every index in every non-empty segment,
    /// in order.  This avoids the per-increment branch of the chaining
    /// iterator and compiles to N separate tight loops.
    template <typename Fn>
    constexpr void for_each(Fn &&fn) const {
        for_each_impl(std::forward<Fn>(fn),
                      std::index_sequence_for<Ranges...>{});
    }

   private:
    template <std::size_t... Is>
    constexpr auto contains_impl(index_type idx,
                                  std::index_sequence<Is...>) const -> bool {
        return (std::get<Is>(segments).contains(idx) || ...);
    }

    template <std::size_t... Is>
    constexpr auto size_impl(std::index_sequence<Is...>) const -> index_type {
        return (std::get<Is>(segments).size() + ...);
    }

    template <std::size_t... Is>
    constexpr auto empty_impl(std::index_sequence<Is...>) const -> bool {
        return (std::get<Is>(segments).empty() && ...);
    }

    template <typename Fn, std::size_t... Is>
    constexpr void for_each_impl(Fn &&fn,
                                  std::index_sequence<Is...>) const {
        (([&] {
            for (auto j : std::get<Is>(segments)) {
                fn(j);
            }
        }()), ...);
    }
};

// CTAD guide: deduce Ranges from constructor arguments.
template <IndexSet... Rs>
DisjointRange(std::tuple<Rs...>) -> DisjointRange<Rs...>;

static_assert(IndexSet<DisjointRange<ContiguousIndexRange>>);
static_assert(IndexSet<
    DisjointRange<ContiguousIndexRange, ContiguousIndexRange>>);

// ─────────────────────────────────────────────────────────────────────────────
/// @brief Concept to detect types with a for_each method.
///
/// Used by MatrixProduct / MatrixVectorProduct to select the zero-overhead
/// compile-time-unrolled path when available.
// ─────────────────────────────────────────────────────────────────────────────
template <typename R>
concept HasForEach = requires(const R &r) {
    r.for_each([](index_type) {});
};

// ─────────────────────────────────────────────────────────────────────────────
/// @brief Merge-sort a fixed-size array of ContiguousIndexRange segments,
///        merging overlapping/adjacent segments in-place.
///
/// After merging, unused slots are set to empty {0, 0} and moved to the end.
// ─────────────────────────────────────────────────────────────────────────────
template <std::size_t N>
constexpr void merge_segments(std::array<ContiguousIndexRange, N> &segs) {
    // Sort by .first (simple insertion sort for small N).
    for (std::size_t i = 1; i < N; ++i) {
        for (std::size_t j = i;
             j > 0 && segs[j].first < segs[j - 1].first; --j) {
            std::swap(segs[j], segs[j - 1]);
        }
    }

    // Merge overlapping/adjacent.
    std::size_t write = 0;
    for (std::size_t read = 0; read < N; ++read) {
        if (segs[read].empty()) continue;
        if (write > 0 && segs[read].first <= segs[write - 1].last) {
            // Merge with previous.
            segs[write - 1].last =
                std::max(segs[write - 1].last, segs[read].last);
        } else {
            segs[write++] = segs[read];
        }
    }
    // Clear unused slots.
    for (std::size_t i = write; i < N; ++i) {
        segs[i] = {index_type{0}, index_type{0}};
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// range_union() overloads
//
// Compute the set union of two IndexSet values.  The result is a
// DisjointRange whose segments are sorted, non-overlapping, and cover
// exactly the union of the input index sets.
// ─────────────────────────────────────────────────────────────────────────────

/// Helper: build a DisjointRange from a fixed-size array of CRs.
template <std::size_t... Is>
constexpr auto make_disjoint_from_array(
    const std::array<ContiguousIndexRange, sizeof...(Is)> &arr,
    std::index_sequence<Is...>) {
    return DisjointRange<decltype((void(Is), ContiguousIndexRange{}))...>{
        std::tuple{arr[Is]...}};
}

template <std::size_t N>
constexpr auto make_disjoint_from_array(
    const std::array<ContiguousIndexRange, N> &arr) {
    return make_disjoint_from_array(arr, std::make_index_sequence<N>{});
}

/// @brief Union of two ContiguousIndexRanges.
///
/// Returns DisjointRange<CR, CR>.  If the inputs overlap or are adjacent,
/// one segment holds the merged range and the other is empty.
constexpr auto range_union(const ContiguousIndexRange &a,
                           const ContiguousIndexRange &b) {
    std::array<ContiguousIndexRange, 2> segs{a, b};
    merge_segments(segs);
    return make_disjoint_from_array(segs);
}

// ── EmptyIndexRange is the identity element for union ────────────────────────

/// @brief Empty ∪ Empty → Empty.
constexpr auto range_union([[maybe_unused]] const EmptyIndexRange &,
                           [[maybe_unused]] const EmptyIndexRange &)
    -> EmptyIndexRange {
    return {};
}

/// @brief Empty ∪ R → R.
template <IndexSet R>
    requires(!IsEmptyIndexRange<R>)
constexpr auto range_union([[maybe_unused]] const EmptyIndexRange &,
                           const R &r) {
    return r;
}

/// @brief R ∪ Empty → R.
template <IndexSet R>
    requires(!IsEmptyIndexRange<R>)
constexpr auto range_union(const R &r,
                           [[maybe_unused]] const EmptyIndexRange &) {
    return r;
}

/// @brief Union of a DisjointRange with a ContiguousIndexRange.
template <IndexSet... As>
constexpr auto range_union(const DisjointRange<As...> &dr,
                           const ContiguousIndexRange &b) {
    constexpr std::size_t M = sizeof...(As) + 1;
    std::array<ContiguousIndexRange, M> segs{};
    fill_segments_from_disjoint(dr, segs, std::index_sequence_for<As...>{});
    segs[sizeof...(As)] = b;
    merge_segments(segs);
    return make_disjoint_from_array(segs);
}

/// @brief Union of a ContiguousIndexRange with a DisjointRange.
template <IndexSet... Bs>
constexpr auto range_union(const ContiguousIndexRange &a,
                           const DisjointRange<Bs...> &dr) {
    return range_union(dr, a);
}

/// @brief Union of two DisjointRanges.
template <IndexSet... As, IndexSet... Bs>
constexpr auto range_union(const DisjointRange<As...> &a,
                           const DisjointRange<Bs...> &b) {
    constexpr std::size_t M = sizeof...(As) + sizeof...(Bs);
    std::array<ContiguousIndexRange, M> segs{};
    fill_segments_from_disjoint(a, segs, std::index_sequence_for<As...>{});
    fill_segments_from_disjoint_offset<sizeof...(As)>(
        b, segs, std::index_sequence_for<Bs...>{});
    merge_segments(segs);
    return make_disjoint_from_array(segs);
}

/// @brief Union where one or both sides are non-CR/non-DR IndexSets.
///
/// Promotes to ContiguousIndexRange first, then dispatches.
template <IndexSet A, IndexSet B>
    requires(!std::is_same_v<std::remove_cvref_t<A>, ContiguousIndexRange> ||
             !std::is_same_v<std::remove_cvref_t<B>, ContiguousIndexRange>) &&
            (!IsDisjointRange<A>) && (!IsDisjointRange<B>)
constexpr auto range_union(const A &a, const B &b) {
    return range_union(to_contiguous_range(a), to_contiguous_range(b));
}

/// @brief Union of a non-CR/non-DR with a DisjointRange.
template <IndexSet A, IndexSet... Bs>
    requires(!std::is_same_v<std::remove_cvref_t<A>,
                              ContiguousIndexRange>) &&
            (!IsDisjointRange<A>)
constexpr auto range_union(const A &a, const DisjointRange<Bs...> &b) {
    return range_union(to_contiguous_range(a), b);
}

/// @brief Union of a DisjointRange with a non-CR/non-DR.
template <IndexSet... As, IndexSet B>
    requires(!std::is_same_v<std::remove_cvref_t<B>,
                              ContiguousIndexRange>) &&
            (!IsDisjointRange<B>)
constexpr auto range_union(const DisjointRange<As...> &a, const B &b) {
    return range_union(a, to_contiguous_range(b));
}

// ── Helpers for extracting segments into arrays ──────────────────────────────

template <IndexSet... Rs, std::size_t N, std::size_t... Is>
constexpr void fill_segments_from_disjoint(
    const DisjointRange<Rs...> &dr,
    std::array<ContiguousIndexRange, N> &out,
    std::index_sequence<Is...>) {
    ((out[Is] = to_contiguous_range(std::get<Is>(dr.segments))), ...);
}

template <std::size_t Offset, IndexSet... Rs, std::size_t N,
          std::size_t... Is>
constexpr void fill_segments_from_disjoint_offset(
    const DisjointRange<Rs...> &dr,
    std::array<ContiguousIndexRange, N> &out,
    std::index_sequence<Is...>) {
    ((out[Offset + Is] = to_contiguous_range(std::get<Is>(dr.segments))),
     ...);
}

// ═════════════════════════════════════════════════════════════════════════════
// to_index_set() — bridge from mdspan slice types to IndexSet types
//
// Converts mdspan-compatible slice descriptors into the corresponding
// IndexSet type for use in sparsity analysis and set operations.
//
//   full_extent_t              → FullRange{extent}
//   integer index              → SingleIndexRange{idx}
//   strided_slice{o,e,s}       → StridedIndexSet{first, last, stride}
//
// All overloads take the dimension extent as a second parameter for
// uniform calling convention (full_extent_t needs it, others ignore it).
// ═════════════════════════════════════════════════════════════════════════════

/// @brief Convert full_extent_t to a FullRange covering [0, extent).
constexpr auto to_index_set([[maybe_unused]] const full_extent_t &,
                            index_type extent) -> FullRange {
    return FullRange{extent};
}

/// @brief Convert an integer index to a SingleIndexRange.
constexpr auto to_index_set(index_type idx,
                            [[maybe_unused]] index_type extent)
    -> SingleIndexRange {
    return SingleIndexRange{idx};
}

/// @brief Convert a strided_slice to a StridedIndexSet.
///
/// A strided_slice{offset, extent, stride} accesses child indices:
///   {offset, offset+stride, offset+2*stride, ...}
/// with ceil(extent/stride) elements.  We convert to StridedIndexSet
/// with first=offset, last=offset+(count-1)*stride+1 (exclusive upper
/// bound), stride=stride.
///
/// When stride is 1, the result is a ContiguousIndexSet (via the type
/// alias, since StridedIndexSet<..., static_index_t<1>> = ContiguousIndexSet).
template <typename OffsetType, typename ExtentType, typename StrideType>
constexpr auto to_index_set(
    const strided_slice<OffsetType, ExtentType, StrideType> &s,
    [[maybe_unused]] index_type extent) {
    const auto o = index_type(s.offset);
    const auto e = index_type(s.extent);
    const auto st = index_type(s.stride);
    if (e == 0 || st == 0) {
        return StridedIndexRange{index_type{0}, index_type{0}, st};
    }
    // Number of output elements: ceil(e / st)
    const auto count = 1 + (e - 1) / st;
    // last = one past the last accessed child index
    const auto last = o + (count - 1) * st + 1;
    return StridedIndexRange{o, last, st};
}

// ═════════════════════════════════════════════════════════════════════════════
// range_intersection() — compute set intersection of two IndexSet values
//
// Overloads are ordered by specificity to avoid ambiguity:
//   0. EmptyIndexRange ∩ R  → EmptyIndexRange  (absorbing element)
//   1. FullIndexSet ∩ R  → R  (identity element)
//   2. SingleIndexSet ∩ R → 0-or-1-element ContiguousIndexRange
//   3. ContiguousIndexSet ∩ ContiguousIndexSet → ContiguousIndexRange
//   4. StridedIndexSet ∩ ContiguousIndexSet → StridedIndexRange
//   5. DisjointRange ∩ R → (recursive per-segment intersection)
//   6. Generic fallback → DynamicSparseIndexSet (iterate + filter)
// ═════════════════════════════════════════════════════════════════════════════

// ── 0. EmptyIndexRange is the absorbing element for intersection ─────────────

/// @brief Empty ∩ Empty → Empty.
constexpr auto range_intersection([[maybe_unused]] const EmptyIndexRange &,
                                  [[maybe_unused]] const EmptyIndexRange &)
    -> EmptyIndexRange {
    return {};
}

/// @brief Empty ∩ R → Empty.
template <IndexSet R>
    requires(!IsEmptyIndexRange<R>)
constexpr auto range_intersection([[maybe_unused]] const EmptyIndexRange &,
                                  [[maybe_unused]] const R &)
    -> EmptyIndexRange {
    return {};
}

/// @brief R ∩ Empty → Empty.
template <IndexSet R>
    requires(!IsEmptyIndexRange<R>)
constexpr auto range_intersection([[maybe_unused]] const R &,
                                  [[maybe_unused]] const EmptyIndexRange &)
    -> EmptyIndexRange {
    return {};
}

// ── 1. FullIndexSet is the identity for intersection ─────────────────────────

/// @brief Full ∩ Full → ContiguousIndexRange{0, min(a.extent, b.extent)}.
template <typename EA, typename EB>
constexpr auto range_intersection(const FullIndexSet<EA> &a,
                                  const FullIndexSet<EB> &b)
    -> ContiguousIndexRange {
    return {index_type{0},
            std::min(index_type(a.extent), index_type(b.extent))};
}

/// @brief Full ∩ R → R (as a contiguous range, preserving the other side).
template <IsFullIndexSet F, IndexSet R>
    requires(!IsFullIndexSet<R>)
constexpr auto range_intersection([[maybe_unused]] const F &,
                                  const R &r) {
    return r;
}

/// @brief R ∩ Full → R (symmetric).
template <IndexSet R, IsFullIndexSet F>
    requires(!IsFullIndexSet<R>)
constexpr auto range_intersection(const R &r,
                                  [[maybe_unused]] const F &) {
    return r;
}

// ── 2. SingleIndexSet ∩ R ────────────────────────────────────────────────────

/// @brief Single ∩ Single → 0-or-1-element range.
template <typename VA, typename VB>
constexpr auto range_intersection(const SingleIndexSet<VA> &a,
                                  const SingleIndexSet<VB> &b)
    -> ContiguousIndexRange {
    if (index_type(a.value) == index_type(b.value)) {
        return {index_type(a.value), index_type(a.value) + 1};
    }
    return {index_type{0}, index_type{0}};
}

/// @brief Single ∩ R → check containment, return 0-or-1-element range.
template <IsSingleIndexSet S, IndexSet R>
    requires(!IsFullIndexSet<R>) && (!IsSingleIndexSet<R>)
constexpr auto range_intersection(const S &s, const R &r)
    -> ContiguousIndexRange {
    if (r.contains(index_type(s.value))) {
        return {index_type(s.value), index_type(s.value) + 1};
    }
    return {index_type{0}, index_type{0}};
}

/// @brief R ∩ Single → symmetric.
template <IndexSet R, IsSingleIndexSet S>
    requires(!IsFullIndexSet<R>) && (!IsSingleIndexSet<R>)
constexpr auto range_intersection(const R &r, const S &s)
    -> ContiguousIndexRange {
    return range_intersection(s, r);
}

// ── 3. ContiguousIndexSet ∩ ContiguousIndexSet ───────────────────────────────

/// @brief Contiguous ∩ Contiguous → max of firsts, min of lasts.
template <typename FA, typename LA, typename FB, typename LB>
constexpr auto range_intersection(
    const ContiguousIndexSet<FA, LA> &a,
    const ContiguousIndexSet<FB, LB> &b) -> ContiguousIndexRange {
    const auto f = std::max(index_type(a.first), index_type(b.first));
    const auto l = std::min(index_type(a.last), index_type(b.last));
    if (f >= l) return {index_type{0}, index_type{0}};
    return {f, l};
}

// ── 4. StridedIndexSet ∩ ContiguousIndexSet ──────────────────────────────────

/// @brief Strided ∩ Contiguous → clamp strided bounds to contiguous window.
///
/// Given StridedIndexSet{f, l, s} and ContiguousIndexSet{a, b}:
///   new_first = smallest element of the strided set that is >= a
///   new_last  = min(l, b)
/// Result is StridedIndexRange{new_first, new_last, s}.
template <typename FT, typename LT, typename ST, typename FA, typename LA>
    requires(!IsContiguousIndexSet<StridedIndexSet<FT, LT, ST>>)
constexpr auto range_intersection(
    const StridedIndexSet<FT, LT, ST> &strided,
    const ContiguousIndexSet<FA, LA> &contig) -> StridedIndexRange {
    const auto f = index_type(strided.first);
    const auto l = index_type(strided.last);
    const auto s = index_type(strided.stride);
    const auto a = index_type(contig.first);
    const auto b = index_type(contig.last);

    if (f >= l || a >= b || s == 0) return {index_type{0}, index_type{0}, s};

    // Find first strided element >= a.
    index_type new_first;
    if (a <= f) {
        new_first = f;
    } else {
        // Smallest k such that f + k*s >= a  →  k = ceil((a - f) / s)
        const auto diff = a - f;
        const auto k = (diff + s - 1) / s;
        new_first = f + k * s;
    }

    const auto new_last = std::min(l, b);
    if (new_first >= new_last) return {index_type{0}, index_type{0}, s};
    return {new_first, new_last, s};
}

/// @brief Contiguous ∩ Strided → symmetric.
template <typename FA, typename LA, typename FT, typename LT, typename ST>
    requires(!IsContiguousIndexSet<StridedIndexSet<FT, LT, ST>>)
constexpr auto range_intersection(
    const ContiguousIndexSet<FA, LA> &contig,
    const StridedIndexSet<FT, LT, ST> &strided) -> StridedIndexRange {
    return range_intersection(strided, contig);
}

// ── 5. DisjointRange ∩ R ────────────────────────────────────────────────────

// Forward declaration for the recursive case.
template <IndexSet A, IndexSet B>
    requires(!IsFullIndexSet<A>) && (!IsSingleIndexSet<A>) &&
            (!IsFullIndexSet<B>) && (!IsSingleIndexSet<B>) &&
            (!IsDisjointRange<A>) && (!IsDisjointRange<B>) &&
            // Exclude the cases already handled by overloads 3 and 4
            (!(IsContiguousIndexSet<A> && IsContiguousIndexSet<B>)) &&
            (!((IsStridedIndexSet<A> && !IsContiguousIndexSet<A>) &&
               IsContiguousIndexSet<B>)) &&
            (!((IsStridedIndexSet<B> && !IsContiguousIndexSet<B>) &&
               IsContiguousIndexSet<A>))
auto range_intersection(const A &a, const B &b) -> DynamicSparseIndexSet;

/// @brief DisjointRange ∩ IndexSet → intersect each segment, collect results.
///
/// Returns a DynamicSparseIndexSet containing the union of per-segment
/// intersections.
template <IndexSet... As, IndexSet R>
    requires(!IsFullIndexSet<R>) && (!IsSingleIndexSet<R>) &&
            (!IsDisjointRange<R>)
auto range_intersection(const DisjointRange<As...> &dr, const R &r)
    -> DynamicSparseIndexSet {
    DynamicSparseIndexSet result{{}};
    auto process = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (([&] {
            auto seg_result =
                range_intersection(std::get<Is>(dr.segments), r);
            for (auto it = seg_result.begin(); it != seg_result.end();
                 ++it) {
                result.indices.push_back(*it);
            }
        }()), ...);
    };
    process(std::index_sequence_for<As...>{});
    // Result should already be sorted since segments are ordered and
    // non-overlapping. Deduplicate just in case.
    auto last = std::unique(result.indices.begin(), result.indices.end());
    result.indices.erase(last, result.indices.end());
    return result;
}

/// @brief IndexSet ∩ DisjointRange → symmetric.
template <IndexSet R, IndexSet... Bs>
    requires(!IsFullIndexSet<R>) && (!IsSingleIndexSet<R>) &&
            (!IsDisjointRange<R>)
auto range_intersection(const R &r, const DisjointRange<Bs...> &dr)
    -> DynamicSparseIndexSet {
    return range_intersection(dr, r);
}

/// @brief DisjointRange ∩ DisjointRange → pairwise segment intersections.
template <IndexSet... As, IndexSet... Bs>
auto range_intersection(const DisjointRange<As...> &a,
                        const DisjointRange<Bs...> &b)
    -> DynamicSparseIndexSet {
    DynamicSparseIndexSet result{{}};
    auto process_a = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (([&] {
            auto seg_result =
                range_intersection(std::get<Is>(a.segments), b);
            for (auto idx : seg_result.indices) {
                result.indices.push_back(idx);
            }
        }()), ...);
    };
    process_a(std::index_sequence_for<As...>{});
    // Sort + deduplicate (pairwise intersections preserve segment order
    // but we collect across multiple a-segments).
    std::sort(result.indices.begin(), result.indices.end());
    auto last = std::unique(result.indices.begin(), result.indices.end());
    result.indices.erase(last, result.indices.end());
    return result;
}

// ── 6. Generic fallback → DynamicSparseIndexSet ─────────────────────────────

/// @brief Generic fallback: iterate the smaller set, filter with contains().
///
/// This handles any pair of IndexSet types not covered by the
/// specialized overloads above (e.g., Strided ∩ Strided, Sparse ∩ Strided).
template <IndexSet A, IndexSet B>
    requires(!IsFullIndexSet<A>) && (!IsSingleIndexSet<A>) &&
            (!IsFullIndexSet<B>) && (!IsSingleIndexSet<B>) &&
            (!IsDisjointRange<A>) && (!IsDisjointRange<B>) &&
            // Exclude the cases already handled by overloads 3 and 4
            (!(IsContiguousIndexSet<A> && IsContiguousIndexSet<B>)) &&
            (!((IsStridedIndexSet<A> && !IsContiguousIndexSet<A>) &&
               IsContiguousIndexSet<B>)) &&
            (!((IsStridedIndexSet<B> && !IsContiguousIndexSet<B>) &&
               IsContiguousIndexSet<A>))
auto range_intersection(const A &a, const B &b) -> DynamicSparseIndexSet {
    DynamicSparseIndexSet result{{}};
    // Iterate over the smaller set, check containment in the larger.
    if (a.size() <= b.size()) {
        for (auto it = a.begin(); it != a.end(); ++it) {
            if (b.contains(*it)) {
                result.indices.push_back(*it);
            }
        }
    } else {
        for (auto it = b.begin(); it != b.end(); ++it) {
            if (a.contains(*it)) {
                result.indices.push_back(*it);
            }
        }
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
/// @brief Trait to mark unary functors that preserve zero: Op(0) == 0.
///
/// Specialize `is_zero_preserving_unary_op<Op>` for types where this holds.
/// By default, `std::negate` is zero-preserving.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Op>
struct is_zero_preserving_unary_op : std::false_type {};

// std::negate preserves zero: -0 == 0
template <typename T>
struct is_zero_preserving_unary_op<std::negate<T>> : std::true_type {};

template <typename Op>
concept ZeroPreservingUnaryOp = is_zero_preserving_unary_op<Op>::value;

// ─────────────────────────────────────────────────────────────────────────────
/// @brief Trait to mark binary scalar functors that preserve zero when
///        applied with a fixed scalar: Op(0, s) == 0 or Op(s, 0) == 0.
///
/// The template parameters are:
///   - Op: the binary functor type
///   - ScalarOnRight: true if the scalar is the second argument
///
/// By default, `std::multiplies` and `std::divides` (with scalar on right)
/// are zero-preserving.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Op, bool ScalarOnRight>
struct is_zero_preserving_scalar_op : std::false_type {};

// multiplies: 0 * s == 0, s * 0 == 0
template <typename T, bool ScalarOnRight>
struct is_zero_preserving_scalar_op<std::multiplies<T>, ScalarOnRight>
    : std::true_type {};

// divides: 0 / s == 0 (scalar on right only)
template <typename T>
struct is_zero_preserving_scalar_op<std::divides<T>, /*ScalarOnRight=*/true>
    : std::true_type {};

template <typename Op, bool ScalarOnRight>
concept ZeroPreservingScalarOp =
    is_zero_preserving_scalar_op<Op, ScalarOnRight>::value;

}  // namespace zipper::expression::detail
