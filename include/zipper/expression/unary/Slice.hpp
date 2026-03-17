#if !defined(ZIPPER_EXPRESSION_UNARY_SLICE_HPP)
#define ZIPPER_EXPRESSION_UNARY_SLICE_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/IndexSlice.hpp"
#include <utility>
#include "zipper/concepts/Expression.hpp"
#include "zipper/concepts/Index.hpp"
#include "zipper/concepts/stl.hpp"
#include "zipper/detail/constexpr_arithmetic.hpp"
#include "zipper/detail/is_integral_constant.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"
#include "zipper/expression/detail/IndexSet.hpp"

namespace zipper::expression {
namespace unary {
template <zipper::concepts::QualifiedExpression ExprType,
          typename... Slices>
class Slice;

namespace _detail_slice {
template <typename T>
struct slice_helper;

template <typename OffsetType, typename ExtentType, typename StrideType>
struct slice_helper<strided_slice<OffsetType, ExtentType, StrideType>> {
   public:
    using type = strided_slice<OffsetType, ExtentType, StrideType>;
    constexpr static bool stores_references = false;
    constexpr slice_helper(const type &t) : m_slice(t) {}
    constexpr static auto get_extent(const auto &stride, const auto &extent) {
        return extent > 0 ? 1 + (extent - 1) / stride : 0;
    }

    template <rank_type N, zipper::concepts::Extents ET>
    constexpr static index_type static_extent() {
        if constexpr (zipper::detail::is_integral_constant_v<StrideType> &&
                      zipper::detail::is_integral_constant_v<ExtentType>) {
            return get_extent(StrideType::value, ExtentType::value);
        } else {
            return std::dynamic_extent;
        }
    }

    template <rank_type N, zipper::concepts::Extents ET>
    static auto internal_extent(const type &t, const ET &e) {
        if constexpr (std::is_same_v<ExtentType,
                                     static_index_t<std::dynamic_extent>>) {
            return e.extent(N);
        } else {
            return t.extent;
        }
    }

    template <rank_type N, zipper::concepts::Extents ET>
    constexpr static index_type extent(const type &t, const ET &ex) {
        zipper::detail::ConstexprArithmetic s = t.stride;
        zipper::detail::ConstexprArithmetic e = internal_extent<N>(t, ex);
        return get_extent(s, e);
    }

    index_type get_index(index_type input) const {
        index_type start = zipper::detail::first_of(m_slice);
        index_type stride = zipper::detail::stride_of(m_slice);
        return start + input * stride;
    }

    constexpr auto slice() const -> const type & { return m_slice; }
    constexpr auto make_owned() const -> slice_helper { return *this; }

   private:
    type m_slice;
};
template <>
struct slice_helper<full_extent_t> {
   public:
    using type = full_extent_t;
    constexpr static bool stores_references = false;
    constexpr slice_helper(const type &) {}
    template <rank_type N, zipper::concepts::Extents ET>
    constexpr static index_type static_extent() {
        return ET::static_extent(N);
    }
    template <rank_type N, zipper::concepts::Extents ET>
    constexpr static index_type extent(const type &, const ET &e) {
        return e.extent(N);
    }

    constexpr index_type get_index(index_type input) const { return input; }

    constexpr auto slice() const -> type { return full_extent_t{}; }
    constexpr auto make_owned() const -> slice_helper { return *this; }
};
template <typename T>
    requires(std::is_integral_v<T>)
struct slice_helper<T> {
   public:
    using type = index_type;
    constexpr static bool stores_references = false;
    constexpr slice_helper(const type &t) : m_slice(t) {}
    template <rank_type N, zipper::concepts::Extents ET>
    constexpr static index_type static_extent() {
        return 1;
    }
    template <rank_type N, zipper::concepts::Extents ET>
    constexpr static index_type extent(const type & /*t*/, const ET &) {
        return static_extent<N, ET>();
    }

    constexpr index_type get_index() const { return m_slice; }

    constexpr auto slice() const -> const type & { return m_slice; }
    constexpr auto make_owned() const -> slice_helper { return *this; }

   private:
    index_type m_slice;
};

template <index_type N>
struct slice_helper<std::integral_constant<index_type, N>> {
   public:
    using type = std::integral_constant<index_type, N>;
    constexpr static bool stores_references = false;
    constexpr slice_helper(std::integral_constant<index_type, N>) {}
    template <rank_type M, zipper::concepts::Extents ET>
    constexpr static index_type static_extent() {
        return 1;
    }
    template <rank_type M, zipper::concepts::Extents ET>
    constexpr static index_type extent(const type &, const ET &) {
        return static_extent<M, ET>();
    }

    constexpr index_type get_index() const { return N; }

    constexpr auto slice() const -> type { return type{}; }
    constexpr auto make_owned() const -> slice_helper { return *this; }
};
template <zipper::concepts::QualifiedExpression Expr>
    requires(Expr::extents_type::rank() == 1)
struct slice_helper<Expr> {
   public:
    using type = Expr;
    using expr_traits =
        zipper::expression::detail::ExpressionTraits<std::decay_t<Expr>>;
    /// The stored expression may itself hold references to external data.
    constexpr static bool stores_references = expr_traits::stores_references;
    constexpr slice_helper(const type &t) : m_slice(t) {}
    template <rank_type M, zipper::concepts::Extents ET>
    constexpr static index_type static_extent() {
        return Expr::extents_type::static_extent(0);
    }
    template <rank_type M, zipper::concepts::Extents ET>
    constexpr static index_type extent(const type &t, const ET &) {
        return t.extent(0);
    }

    constexpr index_type get_index(index_type input) const {
        return m_slice(input);
    }

    constexpr auto slice() const -> const type & { return m_slice; }

    /// Deep-copy the stored expression so the result owns all data.
    auto make_owned() const {
        auto owned_expr = m_slice.make_owned();
        return slice_helper<decltype(owned_expr)>(std::move(owned_expr));
    }

   private:
    type m_slice;
};

// Note: A slice_helper<Wrapper> specialization for ZipperBase-derived types
// is no longer needed. ZipperBase::slice_expression() now unwraps wrapper
// types at the type level via detail::slice_type_for_t, so slice_helper
// always receives the raw expression type (handled by the QualifiedExpression
// specialization above).

template <size_t N>
struct slice_helper<std::array<index_type, N>> {
   public:
    using type = std::array<index_type, N>;
    constexpr static bool stores_references = false;
    constexpr slice_helper(const type &t) : m_slice(t) {}
    template <rank_type M, zipper::concepts::Extents ET>
    constexpr static index_type static_extent() {
        return index_type(N);
    }
    template <rank_type M, zipper::concepts::Extents ET>
    constexpr static index_type extent(const type & /*t*/, const ET &) {
        return static_extent<M, ET>();
    }

    constexpr index_type get_index(index_type input) const {
        return m_slice[input];
    }

    constexpr auto slice() const -> const type & { return m_slice; }
    constexpr auto make_owned() const -> slice_helper { return *this; }

   private:
    type m_slice;
};
template <>
struct slice_helper<std::vector<index_type>> {
   public:
    using type = std::vector<index_type>;
    constexpr static bool stores_references = false;
    constexpr slice_helper(const type &t) : m_slice(t) {}
    template <rank_type N, zipper::concepts::Extents ET>
    constexpr static index_type static_extent() {
        return std::dynamic_extent;
    }
    template <rank_type N, zipper::concepts::Extents ET>
    constexpr static index_type extent(const type &t, const ET &) {
        return t.size();
    }

    constexpr index_type get_index(index_type input) const {
        return m_slice[input];
    }

    constexpr auto slice() const -> const type & { return m_slice; }
    constexpr auto make_owned() const -> slice_helper { return *this; }

   private:
    std::vector<index_type> m_slice;
};

template <zipper::concepts::Extents, typename, typename...>
struct _slice_extent_helper;
template <zipper::concepts::Extents ET, rank_type... N, typename... Slices>
struct _slice_extent_helper<ET, std::integer_sequence<rank_type, N...>,
                            Slices...> {
    template <rank_type R>
    using get_slice_t =
        std::decay_t<std::tuple_element_t<R, std::tuple<Slices...>>>;

    template <size_t M>
    using single_slice_helper = slice_helper<get_slice_t<M>>;

    template <std::size_t... Indices>
    consteval static std::array<rank_type, sizeof...(Indices)> get_actionable(
        std::index_sequence<Indices...>)
        requires(sizeof...(Indices) == ET::rank())
    {
        constexpr size_t Rank = sizeof...(Indices);
        using tuple_type = std::tuple<Slices...>;
        constexpr auto eval =
            []<rank_type J>(std::integral_constant<rank_type, J>) -> rank_type {
            if (zipper::concepts::IndexSlice<
                    std::tuple_element_t<J, tuple_type>> &&
                !zipper::concepts::Index<
                    std::tuple_element_t<J, tuple_type>>) {
                return 0;
            } else {
                return std::dynamic_extent;
            }
        };
        std::array<rank_type, Rank> r{
            {eval(std::integral_constant<size_t, Indices>{})...}};
        rank_type index = 0;
        for (auto &v : r) {
            if (v == 0) {
                v = index++;
            }
        }
        return r;
    }

    constexpr static auto actionable_indices =
        get_actionable(std::make_index_sequence<ET::rank()>{});
    constexpr static rank_type rank = []() {
        rank_type j = 0;
        for (const auto &i : actionable_indices) {
            if (i != std::dynamic_extent) {
                j++;
            }
        }
        return j;
    }();
    template <typename>
    struct _extents_maker;

    template <rank_type... Rs>
    struct _extents_maker<std::integer_sequence<rank_type, Rs...>> {
        consteval static rank_type get_rank(rank_type r) {
            for (rank_type j = 0; j < actionable_indices.size(); ++j) {
                if (actionable_indices[j] == r) {
                    return j;
                }
            }
            // clang doesn't catch that all executions of this consteval turn
            // out to return
            std::unreachable();
            return rank + 1;
        }
        consteval static rank_type get_dynamic_rank(rank_type r) {
            return get_rank(dynamic_indices[r]);
        }
        using type = extents<single_slice_helper<get_rank(
            Rs)>::template static_extent<get_rank(Rs), ET>()...>;

        constexpr static std::array<rank_type, type::rank_dynamic()>
            dynamic_indices =
                zipper::detail::extents::dynamic_extents_indices_v<type>;

        template <typename>
        struct _dynamic_helper;
        template <rank_type... Ds>
        struct _dynamic_helper<std::integer_sequence<rank_type, Ds...>> {
            constexpr static auto get_extents(const ET &base,
                                              const Slices &...slices) -> type {
                return type{
                    single_slice_helper<get_dynamic_rank(Ds)>::template extent<
                        get_dynamic_rank(Ds), ET>(
                        zipper::detail::pack_index<get_dynamic_rank(Ds)>(
                            slices...),
                        base)...

                };
            }
        };
        using dynamic_helper =
            _dynamic_helper<decltype(std::make_integer_sequence<
                                     rank_type, type::rank_dynamic()>{})>;
        constexpr static auto get_extents(const ET &base,
                                          const Slices &...slices) -> type {
            return dynamic_helper::get_extents(base, slices...);
        }
    };
    using maker =
        _extents_maker<decltype(std::make_integer_sequence<rank_type, rank>{})>;
    using extents_type = typename maker::type;

    constexpr static auto get_extents(const ET &base, const Slices &...slices)
        -> extents_type {
        return maker::get_extents(base, slices...);
    }
};
template <zipper::concepts::Extents ET, typename... Slices>
using slice_extent_helper = _slice_extent_helper<
    ET, decltype(std::make_integer_sequence<rank_type, ET::rank()>{}),
    Slices...>;

}  // namespace _detail_slice

}  // namespace unary

/// Implementation details for Slice expressions.
///
/// Holds child traits alias, the slice type lookup template, the extents
/// computation helper, and the actionable-indices map. These are needed by
/// both the ExpressionTraits specialization (to compute extents_type) and
/// the Slice class body (to build extents in the constructor and map
/// output indices to child indices).
template <zipper::concepts::QualifiedExpression ExprType,
          typename... Slices>
struct detail::ExpressionDetail<unary::Slice<ExprType, Slices...>> {
    using Base = detail::ExpressionTraits<std::decay_t<ExprType>>;

    template <rank_type R>
    using get_slice_t =
        std::decay_t<std::tuple_element_t<R, std::tuple<Slices...>>>;

    using extents_helper =
        unary::_detail_slice::slice_extent_helper<typename Base::extents_type,
                                           Slices...>;

    constexpr static std::array<rank_type, Base::extents_type::rank()>
        actionable_indices = extents_helper::get_actionable(
            std::make_index_sequence<Base::extents_type::rank()>{});
};

template <zipper::concepts::QualifiedExpression ExprType,
          typename... Slices>
struct detail::ExpressionTraits<unary::Slice<ExprType, Slices...>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          ExprType> {
    using _Detail = detail::ExpressionDetail<unary::Slice<ExprType, Slices...>>;
    using value_type = typename _Detail::Base::value_type;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;

    using extents_type = typename _Detail::extents_helper::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    /// stores_references accounts for BOTH the main child expression AND
    /// any expression-typed index slices stored in m_slices.  Without this,
    /// a Slice whose main child is owned by value but whose index slice
    /// holds a dangling reference would incorrectly report stores_references
    /// == false.
    constexpr static bool stores_references =
        zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
            ExprType>::stores_references ||
        (unary::_detail_slice::slice_helper<std::decay_t<Slices>>::stores_references || ...);

    /// Propagate has_index_set from child.
    constexpr static bool has_index_set = _Detail::Base::has_index_set;

    /// Backward-compatible alias for has_index_set.
    constexpr static bool has_known_zeros = has_index_set;
};

namespace unary {
template <zipper::concepts::QualifiedExpression ExprType,
          typename... Slices>
class Slice : public UnaryExpressionBase<Slice<ExprType, Slices...>,
                                       ExprType> {
   public:
    using self_type = Slice<ExprType, Slices...>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using detail_type = zipper::expression::detail::ExpressionDetail<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    using Base = UnaryExpressionBase<self_type, ExprType>;
    using Base::extent;
    using Base::expression;
    using ExpressionType = std::decay_t<ExprType>;
    using expr_traits = zipper::expression::detail::ExpressionTraits<ExpressionType>;
    using expr_extents_type = expr_traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    using slice_storage_type =
        std::tuple<_detail_slice::slice_helper<std::decay_t<Slices>>...>;

    constexpr static std::array<rank_type, expr_extents_type::rank()>
        actionable_indices = detail_type::actionable_indices;

    Slice(const Slice &) = default;
    Slice(Slice &&) = default;

    Slice &operator=(zipper::concepts::Expression auto const &v) {
        assign(v);
        return *this;
    }
    template <typename U>
      requires std::constructible_from<typename Base::storage_type, U &&>
    Slice(U &&b, const Slices &...slices)
        : Base(std::forward<U>(b)),
          m_extents(detail_type::extents_helper::get_extents(expression().extents(), slices...)),
          m_slices(_detail_slice::slice_helper<std::decay_t<Slices>>(slices)...) {}

    constexpr auto extent(rank_type i) const -> index_type {
        return m_extents.extent(i);
    }

    constexpr auto extents() const -> extents_type {
        return m_extents;
    }

    template <rank_type K, typename... Args>
    index_type get_index(Args &&...a) const {
        const auto &s = std::get<K>(m_slices);
        constexpr static index_type index = actionable_indices[K];
        if constexpr (index == std::dynamic_extent) {
            return s.get_index();
        } else {
            const auto &v = zipper::detail::pack_index<index>(a...);
            return s.get_index(v);
        }
    }

    template <typename... Args, rank_type... ranks>
    auto _coeff(std::integer_sequence<rank_type, ranks...>,
                Args &&...idxs) const -> value_type {
        return expression().coeff(get_index<ranks>(idxs...)...);
    }
    template <typename... Args, rank_type... ranks>
    auto _coeff_ref(std::integer_sequence<rank_type, ranks...>, Args &&...idxs)
        -> value_type &
        requires(traits::is_assignable())
    {
        return expression().coeff_ref(get_index<ranks>(idxs...)...);
    }
    template <typename... Args, rank_type... ranks>
    auto _const_coeff_ref(std::integer_sequence<rank_type, ranks...>,
                          Args &&...idxs) const -> const value_type &
        requires(traits::is_referrable())
    {
        return expression().const_coeff_ref(get_index<ranks>(idxs...)...);
    }

    template <typename... Args>
    value_type coeff(Args &&...idxs) const {
        return _coeff(
            std::make_integer_sequence<rank_type, expr_extents_type::rank()>{},
            std::forward<Args>(idxs)...);
    }
    template <typename... Args>
    value_type &coeff_ref(Args &&...idxs)
        requires(traits::is_assignable())
    {
        return _coeff_ref(
            std::make_integer_sequence<rank_type, expr_extents_type::rank()>{},
            std::forward<Args>(idxs)...);
    }
    template <typename... Args>
    const value_type &const_coeff_ref(Args &&...idxs) const
        requires(traits::is_referrable())
    {
        return _const_coeff_ref(
            std::make_integer_sequence<rank_type, expr_extents_type::rank()>{},
            std::forward<Args>(idxs)...);
    }

    template <zipper::concepts::Expression V>
    void assign(const V &v)
        requires(
            traits::is_assignable() &&
            extents_traits::template is_convertable_from<
                typename zipper::expression::detail::ExpressionTraits<V>::extents_type>())
    {
        expression::detail::AssignHelper<V, self_type>::assign(v, *this);
    }

    /// Recursively deep-copy child so the result owns all data.
    /// Both the main child expression AND any expression-typed index
    /// slices are recursively owned via their slice_helper::make_owned().
    auto make_owned() const {
        return _make_owned_impl(
            std::make_integer_sequence<rank_type, sizeof...(Slices)>{});
    }

    // ── IndexSet propagation ─────────────────────────────────────────
    //
    // For a Slice on a child with an index set, we propagate the child's
    // index_set by:
    //   1. Mapping the output "other_idx" to the child's index space
    //   2. Querying the child's index_set
    //   3. Remapping the result back to the output index space
    //
    // Supports rank-2 children (the primary use case). Both rank-2 and
    // rank-1 output are handled — for rank-1 output, the collapsed
    // dimension provides the fixed child index automatically.

    // ── Compile-time mapping tables ────────────────────────────────────
    //
    // out_to_child_map[out_dim] = child_dim for each non-collapsed dim.
    // Size = extents_type::rank() (output rank).
    static constexpr auto _out_to_child_map = []() {
      constexpr auto child_rank = expr_extents_type::rank();
      std::array<rank_type, extents_type::rank()> result{};
      for (rank_type c = 0; c < child_rank; ++c) {
        if (actionable_indices[c] != std::dynamic_extent) {
          result[actionable_indices[c]] = c;
        }
      }
      return result;
    }();

    // For each non-collapsed output dim, find the "other" child dim
    // that provides the second coordinate for the child's
    // index_set query. For rank-2 output from rank-2 child:
    // other_child_dim_for[0] = _out_to_child_map[1], etc.
    // For rank-1 output from rank-2 child: the other child dim is
    // the collapsed one.
    static constexpr auto _other_child_dim_for = []() {
      constexpr auto out_rank = extents_type::rank();
      constexpr auto child_rank = expr_extents_type::rank();
      std::array<rank_type, out_rank> result{};
      for (rank_type d = 0; d < out_rank; ++d) {
        rank_type my_child_dim = _out_to_child_map[d];
        // Find any child dim != my_child_dim.  Prefer non-collapsed
        // (i.e., another output dim), but fall back to collapsed.
        result[d] = std::dynamic_extent;  // sentinel
        for (rank_type c = 0; c < child_rank; ++c) {
          if (c != my_child_dim) {
            result[d] = c;
            break;
          }
        }
      }
      return result;
    }();

    /// @brief Returns the index set along output dimension D.
    ///
    /// Maps through the slice to the child's index_set, then
    /// remaps back. Works for both rank-2 and rank-1 output (the
    /// latter from a rank-2 child with one collapsed dimension).
    template <rank_type D>
      requires(traits::has_index_set && D < extents_type::rank())
    auto index_set(index_type other_idx) const {
      return _nonzero_range_impl<D, _out_to_child_map[D],
                                  _other_child_dim_for[D]>(other_idx);
    }

    /// @brief Backward-compatible wrapper; prefer index_set.
    template <rank_type D>
      requires(traits::has_index_set && D < extents_type::rank())
    auto nonzero_range(index_type other_idx) const {
      return index_set<D>(other_idx);
    }

    /// Convenience: col_range_for_row (rank-2 output only).
    auto col_range_for_row(index_type row) const
      requires(traits::has_index_set && extents_type::rank() == 2)
    {
      return index_set<1>(row);
    }

    /// Convenience: row_range_for_col (rank-2 output only).
    auto row_range_for_col(index_type col) const
      requires(traits::has_index_set && extents_type::rank() == 2)
    {
      return index_set<0>(col);
    }

   private:
    /// Core index_set implementation.
    ///
    /// @tparam D          Output dimension being queried
    /// @tparam ChildDimD  Child dimension corresponding to output dim D
    /// @tparam ChildDimOther  The other child dimension (provides the
    ///                        cross-coordinate for the child's query)
    ///
    /// If ChildDimOther is a collapsed dimension (integer slice), its
    /// fixed index is used directly.  Otherwise, other_idx is mapped
    /// through the slice to the child's index space.
    template <rank_type D, rank_type ChildDimD, rank_type ChildDimOther>
    auto _nonzero_range_impl(index_type other_idx) const {
      // Step 1: Map other_idx to the child's index space.
      index_type child_other_idx;
      if constexpr (actionable_indices[ChildDimOther] == std::dynamic_extent) {
        // Collapsed dimension — use the fixed index from the integer slice.
        child_other_idx = std::get<ChildDimOther>(m_slices).get_index();
      } else {
        // Non-collapsed — map through the slice.
        child_other_idx =
            std::get<ChildDimOther>(m_slices).get_index(other_idx);
      }

      // Step 2: Query the child's index set along ChildDimD.
      auto child_range =
          expression().template index_set<ChildDimD>(child_other_idx);

      // Step 3: Remap the child range back to the output index space.
      return _remap_range<ChildDimD>(child_range);
    }

     /// Inverse-map a child-space ContiguousIndexRange through a
    /// strided_slice back to output-space.
    ///
    /// Given child range [cr.first, cr.last) and a slice with
    /// {offset, stride}, computes the output range by:
    ///   out_first = ceil((cr.first - offset) / stride)   (clamped to 0)
    ///   out_last  = ceil((cr.last  - offset) / stride)   (clamped to out_extent)
    ///
    /// Returns empty range if no overlap.
    static auto _inverse_map_strided(
        const zipper::expression::detail::ContiguousIndexRange &cr,
        index_type offset, index_type stride, index_type out_extent)
        -> zipper::expression::detail::ContiguousIndexRange {
      if (cr.last <= offset || cr.first >= offset + out_extent * stride) {
        return {index_type{0}, index_type{0}};
      }
      index_type out_first =
          cr.first <= offset ? index_type{0}
                             : (cr.first - offset + stride - 1) / stride;
      index_type out_last =
          cr.last <= offset ? index_type{0}
                            : std::min((cr.last - offset + stride - 1) / stride,
                                       out_extent);
      return {out_first, out_last};
    }

    /// Remap a child-space index set through the slice for child dimension K
    /// back to the output index space.
    ///
    /// Uses to_contiguous_range() and range_intersection() from the IndexSet
    /// library, then inverse-maps the result.  DisjointRange inputs are
    /// handled by remapping each segment recursively.
    template <rank_type K, typename ChildRange>
    auto _remap_range(const ChildRange &child_range) const {
      using slice_t =
          typename std::tuple_element_t<K, slice_storage_type>::type;

      if constexpr (zipper::expression::detail::IsDisjointRange<ChildRange>) {
        // DisjointRange: remap each segment independently and reassemble.
        return _remap_disjoint<K>(
            child_range,
            std::make_index_sequence<
                std::remove_cvref_t<ChildRange>::num_segments>{});
      } else if constexpr (std::is_same_v<slice_t, full_extent_t>) {
        // Identity mapping — child range IS the output range.
        return zipper::expression::detail::to_contiguous_range(child_range);
      } else if constexpr (requires {
                             std::declval<slice_t>().offset;
                             std::declval<slice_t>().stride;
                           }) {
        // strided_slice — intersect child range with slice's footprint,
        // then inverse-map to output space.
        const auto &s = std::get<K>(m_slices).slice();
        index_type offset = static_cast<index_type>(s.offset);
        index_type stride = static_cast<index_type>(s.stride);
        constexpr rank_type out_dim = actionable_indices[K];
        index_type out_ext = extent(out_dim);

        // Get the child-space footprint of this slice.
        auto slice_set = zipper::expression::detail::to_index_set(
            s, expression().extent(K));
        // Intersect with the child's index set (as contiguous bounding box).
        auto isect = zipper::expression::detail::range_intersection(
            zipper::expression::detail::to_contiguous_range(child_range),
            zipper::expression::detail::to_contiguous_range(slice_set));
        // Inverse-map the intersection back to output space.
        return _inverse_map_strided(isect, offset, stride, out_ext);
      } else {
        // Complex slice (array, vector, expression) — fall back to full range.
        constexpr rank_type out_dim = actionable_indices[K];
        return zipper::expression::detail::ContiguousIndexRange{
            index_type{0}, extent(out_dim)};
      }
    }

    /// Remap each segment of a DisjointRange independently through the
    /// slice, reassembling the results into a new DisjointRange.
    template <rank_type K, typename DR, std::size_t... Is>
    auto _remap_disjoint(const DR &dr, std::index_sequence<Is...>) const {
      // Remap each segment via _remap_range (handles all slice types).
      auto remapped = std::tuple{
          _remap_range<K>(std::get<Is>(dr.segments))...};
      // Build a DisjointRange from the remapped segments.
      return zipper::expression::detail::DisjointRange<
          std::remove_cvref_t<decltype(std::get<Is>(remapped))>...>{remapped};
    }
    template <rank_type... Is>
    auto _make_owned_impl(std::integer_sequence<rank_type, Is...>) const {
        auto owned_child = expression().make_owned();
        // Each slice_helper::make_owned() returns a new slice_helper that
        // owns its data.  For value types (strided_slice, array, etc.) this
        // is a copy.  For expression types, this recursively deep-copies.
        auto owned_slices = std::make_tuple(
            std::get<Is>(m_slices).make_owned()...);
        using owned_tuple_t = decltype(owned_slices);
        return Slice<const decltype(owned_child),
                     typename std::tuple_element_t<Is, owned_tuple_t>::type...>(
            std::move(owned_child),
            std::get<Is>(owned_slices).slice()...);
    }

    extents_type m_extents;
    slice_storage_type m_slices;
};

template <zipper::concepts::Expression ExprType, typename... Slices>
Slice(const ExprType &expr, Slices &&...)
    -> Slice<const ExprType&, std::decay_t<Slices>...>;

template <zipper::concepts::Expression ExprType, typename... Slices>
Slice(ExprType &expr, Slices &&...)
    -> Slice<ExprType&, std::decay_t<Slices>...>;

}  // namespace unary
}  // namespace zipper::expression
#endif
