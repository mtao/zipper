#if !defined(ZIPPER_DETAIL_LAYOUTPREFERENCE_HPP)
#define ZIPPER_DETAIL_LAYOUTPREFERENCE_HPP

#include "zipper/storage/layout_types.hpp"
#include <type_traits>

namespace zipper::detail {

// ── Layout preference types ────────────────────────────────────────────
//
// These tag types propagate through the expression tree to tell eval()
// what kind of result to produce.  Each expression node's
// ExpressionTraits defines `using preferred_layout = ...;` to one of
// these.

/// No preference — eval() will produce a default row-major dense matrix.
struct NoLayoutPreference {};

/// Prefer a dense result with the given layout policy (layout_right =
/// row-major, layout_left = col-major).
template <typename LayoutPolicy>
struct DenseLayoutPreference {
  using layout_policy = LayoutPolicy;
};

/// Prefer a sparse compressed result with the given layout policy
/// (layout_right = CSR, layout_left = CSC).
template <typename LayoutPolicy>
struct SparseLayoutPreference {
  using layout_policy = LayoutPolicy;
};

// ── Convenience aliases ────────────────────────────────────────────────

using PreferRowMajor = DenseLayoutPreference<storage::layout_right>;
using PreferColMajor = DenseLayoutPreference<storage::layout_left>;
using PreferCSR = SparseLayoutPreference<storage::layout_right>;
using PreferCSC = SparseLayoutPreference<storage::layout_left>;

// ── Detection concepts / traits ────────────────────────────────────────

/// True if T is any DenseLayoutPreference instantiation.
template <typename T>
struct is_dense_layout_preference : std::false_type {};
template <typename LP>
struct is_dense_layout_preference<DenseLayoutPreference<LP>> : std::true_type {};
template <typename T>
inline constexpr bool is_dense_layout_preference_v =
    is_dense_layout_preference<T>::value;

/// True if T is any SparseLayoutPreference instantiation.
template <typename T>
struct is_sparse_layout_preference : std::false_type {};
template <typename LP>
struct is_sparse_layout_preference<SparseLayoutPreference<LP>>
    : std::true_type {};
template <typename T>
inline constexpr bool is_sparse_layout_preference_v =
    is_sparse_layout_preference<T>::value;

/// True if T is NoLayoutPreference.
template <typename T>
inline constexpr bool is_no_layout_preference_v =
    std::is_same_v<T, NoLayoutPreference>;

/// True if T has a layout_policy member (dense or sparse preference).
template <typename T>
concept HasLayoutPolicy = requires { typename T::layout_policy; };

// ── FlipLayout ─────────────────────────────────────────────────────────
//
// Swapping row ↔ col for transpose.  Flips layout_right ↔ layout_left
// within both Dense and Sparse preferences.  NoLayoutPreference stays
// unchanged.

namespace _flip_detail {
template <typename LP>
struct flip_layout_policy {
  // layout_right → layout_left, layout_left → layout_right
  using type = std::conditional_t<
      std::is_same_v<LP, storage::layout_right>,
      storage::layout_left,
      storage::layout_right>;
};
template <typename LP>
using flip_layout_policy_t = typename flip_layout_policy<LP>::type;
} // namespace _flip_detail

template <typename Pref>
struct FlipLayout {
  using type = NoLayoutPreference; // fallback
};

template <typename LP>
struct FlipLayout<DenseLayoutPreference<LP>> {
  using type = DenseLayoutPreference<
      typename _flip_detail::flip_layout_policy<LP>::type>;
};

template <typename LP>
struct FlipLayout<SparseLayoutPreference<LP>> {
  using type = SparseLayoutPreference<
      typename _flip_detail::flip_layout_policy<LP>::type>;
};

template <>
struct FlipLayout<NoLayoutPreference> {
  using type = NoLayoutPreference;
};

template <typename Pref>
using flip_layout_t = typename FlipLayout<Pref>::type;

// ── CSC × CSR detection ────────────────────────────────────────────────
//
// The only multiply combination we block: LHS is CSC, RHS is CSR.
// CSR × CSR is fine (row-oriented dot products).
// CSR × CSC is fine (column access on RHS is fast).
// CSC × CSC is fine (column-oriented).
// CSC × CSR is the pathological case: neither operand's layout gives
// an efficient inner-product traversal.

template <typename LhsPref, typename RhsPref>
inline constexpr bool is_csc_times_csr_v =
    std::is_same_v<LhsPref, PreferCSC> &&
    std::is_same_v<RhsPref, PreferCSR>;

// ── Static assertions to verify correctness ────────────────────────────

static_assert(std::is_same_v<flip_layout_t<PreferRowMajor>, PreferColMajor>);
static_assert(std::is_same_v<flip_layout_t<PreferColMajor>, PreferRowMajor>);
static_assert(std::is_same_v<flip_layout_t<PreferCSR>, PreferCSC>);
static_assert(std::is_same_v<flip_layout_t<PreferCSC>, PreferCSR>);
static_assert(std::is_same_v<flip_layout_t<NoLayoutPreference>,
                             NoLayoutPreference>);

static_assert(is_dense_layout_preference_v<PreferRowMajor>);
static_assert(is_dense_layout_preference_v<PreferColMajor>);
static_assert(!is_dense_layout_preference_v<PreferCSR>);
static_assert(!is_dense_layout_preference_v<NoLayoutPreference>);

static_assert(is_sparse_layout_preference_v<PreferCSR>);
static_assert(is_sparse_layout_preference_v<PreferCSC>);
static_assert(!is_sparse_layout_preference_v<PreferRowMajor>);
static_assert(!is_sparse_layout_preference_v<NoLayoutPreference>);

static_assert(is_no_layout_preference_v<NoLayoutPreference>);
static_assert(!is_no_layout_preference_v<PreferCSR>);
static_assert(!is_no_layout_preference_v<PreferRowMajor>);

static_assert(is_csc_times_csr_v<PreferCSC, PreferCSR>);
static_assert(!is_csc_times_csr_v<PreferCSR, PreferCSR>);
static_assert(!is_csc_times_csr_v<PreferCSR, PreferCSC>);
static_assert(!is_csc_times_csr_v<PreferCSC, PreferCSC>);
static_assert(!is_csc_times_csr_v<PreferCSC, NoLayoutPreference>);
static_assert(!is_csc_times_csr_v<NoLayoutPreference, PreferCSR>);
static_assert(!is_csc_times_csr_v<PreferRowMajor, PreferCSR>);

} // namespace zipper::detail

#endif
