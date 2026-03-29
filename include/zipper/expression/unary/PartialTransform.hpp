#if !defined(ZIPPER_EXPRESSION_UNARY_PARTIALTRANSFORM_HPP)
#define ZIPPER_EXPRESSION_UNARY_PARTIALTRANSFORM_HPP

// PartialTransform — fiber-wise function application.
//
// Given a child expression, a function Fn, and a set of fiber dimension
// Indices..., applies Fn to each fiber (slice along the fiber dimensions)
// of the child. The result has the same shape as the child.
//
// For MVP, Fn must return a fiber of the same shape as the input fiber.
// (Future: allow Fn to change the fiber shape, which changes the output shape.)
//
// Example:
//   // Apply a 2x2 Givens rotation to each row of a matrix
//   auto result = A.rowwise().transform([&G](auto row) {
//       return (row * G).eval();
//   });
//
// Efficient assignment: PartialTransform declares
// FiberAssignStrategy<Indices...>, so AssignHelper dispatches to
// assign_to() which calls Fn once per fiber (O(n) calls) instead of
// once per element (O(m*n) calls).

#include "Slice.hpp"
#include "UnaryExpressionBase.hpp"
#include "detail/invert_integer_sequence.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/extents/static_extents_to_array.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/detail/wrap_in_base.hpp"
#include "zipper/expression/detail/AssignStrategy.hpp"
#include "zipper/utils/extents/for_each_index.hpp"

namespace zipper::expression {
namespace unary {

    template <zipper::concepts::QualifiedExpression ExprType,
              typename Fn,
              rank_type... Indices>
    class PartialTransform;

} // namespace unary

// ── ExpressionDetail specialization ────────────────────────────────────

template <zipper::concepts::QualifiedExpression ExprType,
          typename Fn,
          rank_type... Indices>
struct detail::ExpressionDetail<
    unary::PartialTransform<ExprType, Fn, Indices...>> {
    using Base = detail::ExpressionTraits<std::decay_t<ExprType>>;
    using child_extents_type = typename Base::extents_type;
    using index_remover =
        unary::detail::invert_integer_sequence<Base::extents_type::rank(),
                                               Indices...>;
};

// ── ExpressionTraits specialization ────────────────────────────────────

template <zipper::concepts::QualifiedExpression ExprType,
          typename Fn,
          rank_type... Indices>
struct detail::ExpressionTraits<
    unary::PartialTransform<ExprType, Fn, Indices...>>
  : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
        ExprType,
        zipper::detail::AccessFeatures{.is_const = true,
                                       .is_reference = false}> {
    using _Detail = detail::ExpressionDetail<
        unary::PartialTransform<ExprType, Fn, Indices...>>;

    // MVP: output shape = input shape (fn preserves fiber shape).
    using extents_type = typename _Detail::child_extents_type;
    using value_type = typename _Detail::Base::value_type;

    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;

    using assign_strategy = detail::FiberAssignStrategy<Indices...>;
};

namespace unary {

    // ── PartialTransform class ─────────────────────────────────────────────

    /// Indices... are the fiber dimensions (the dimensions the function
    /// operates on). Non-fiber dimensions are iterated over during assignment.
    ///
    /// For a rank-2 matrix:
    ///   - Indices = {0} means column fibers (colwise transform)
    ///   - Indices = {1} means row fibers (rowwise transform)
    template <zipper::concepts::QualifiedExpression ExprType,
              typename Fn,
              rank_type... Indices>
    class PartialTransform
      : public UnaryExpressionBase<PartialTransform<ExprType, Fn, Indices...>,
                                   ExprType> {
      public:
        using self_type = PartialTransform<ExprType, Fn, Indices...>;
        using traits = zipper::expression::detail::ExpressionTraits<self_type>;
        using detail_type =
            zipper::expression::detail::ExpressionDetail<self_type>;
        using extents_type = typename traits::extents_type;
        using value_type = typename traits::value_type;
        using Base = UnaryExpressionBase<self_type, ExprType>;
        using Base::expression;
        using child_extents_type = typename detail_type::child_extents_type;
        using index_remover = typename detail_type::index_remover;

        // ── Non-fiber extents (for iteration in assign_to) ─────────────────

        using non_fiber_extents_type =
            typename index_remover::template assign_types<
                zipper::extents,
                zipper::detail::extents::static_extents_to_array_v<
                    child_extents_type>>;

        // ── Slice type for extracting a fiber ──────────────────────────────
        // At fiber dimensions: full_extent_t. At non-fiber dimensions:
        // index_type.

        template <typename>
        struct slice_type_;
        template <rank_type... N>
        struct slice_type_<std::integer_sequence<rank_type, N...>> {
            // Note: in_sequence returns true for non-fiber indices (indices
            // NOT in the Indices... pack). We want full_extent_t for fiber
            // indices (those IN the pack).
            using type =
                Slice<const std::decay_t<ExprType> &,
                      std::conditional_t<!index_remover::in_sequence(N),
                                         full_extent_t,
                                         index_type>...>;
        };

        using slice_type = typename slice_type_<
            std::decay_t<decltype(std::make_integer_sequence<
                                  rank_type,
                                  child_extents_type::rank()>{})>>::type;

        /// The wrapped type that fn receives — slice wrapped in the
        /// rank-appropriate base (VectorBase for rank-1 fibers, etc.).
        using wrapped_fiber_type = zipper::detail::wrap_in_base_t<slice_type>;

        // ── Construction ───────────────────────────────────────────────────

        template <typename U, typename F>
            requires std::constructible_from<typename Base::storage_type, U &&>
        PartialTransform(U &&expr, F &&fn)
          : Base(std::forward<U>(expr)), m_fn(std::forward<F>(fn)) {}

        PartialTransform() = delete;
        PartialTransform &operator=(const PartialTransform &) = delete;
        PartialTransform &operator=(PartialTransform &&) = delete;
        PartialTransform(PartialTransform &&o) = default;
        PartialTransform(const PartialTransform &o) = default;

        constexpr auto extent(rank_type i) const -> index_type {
            return expression().extent(i);
        }

        constexpr auto extents() const -> extents_type {
            return expression().extents();
        }

        // ── Index mapping helpers ──────────────────────────────────────────

        /// Map a *full* index tuple to the child's Slice argument for
        /// dimension N. Fiber dimensions get full_extent_t, non-fiber
        /// dimensions get the corresponding index from the full tuple.
        /// NOTE: Args... must be the FULL index tuple (one per child dim).
        template <typename... Args, rank_type N>
        auto get_slice_arg_from_full(std::integral_constant<rank_type, N>,
                                     Args &&...idxs) const {
            if constexpr (((N == Indices) || ...)) {
                // This is a fiber dimension -> full_extent_t for the slice
                return zipper::full_extent_t{};
            } else {
                // Index directly by full dimension N in the full tuple
                return zipper::detail::pack_index<N>(
                    std::forward<Args>(idxs)...);
            }
        }

        /// Build a fiber slice given the *full* index tuple.
        /// Fiber dimensions get full_extent_t, non-fiber dimensions get
        /// the index at that position in the tuple.
        template <typename... Args, rank_type... N>
        auto make_fiber_slice_from_full(std::integer_sequence<rank_type, N...>,
                                        Args &&...idxs) const {
            return slice_type(
                expression(),
                get_slice_arg_from_full(std::integral_constant<rank_type, N>{},
                                        std::forward<Args>(idxs)...)...);
        }

        // ── coeff() — slow per-element path ────────────────────────────────
        //
        // For each coefficient request, we construct the entire fiber slice,
        // apply fn, then index into the result. This is O(fiber_size) per
        // call because fn processes the whole fiber. Correct but inefficient
        // when used as a sub-expression; the efficient path is assign_to().

        template <typename... Args, rank_type... N>
        value_type _coeff(std::integer_sequence<rank_type, N...> seq,
                          Args &&...idxs) const
            requires(sizeof...(N) == child_extents_type::rank())
        {
            // Build the fiber slice from the full index tuple
            auto fiber =
                make_fiber_slice_from_full(seq, std::forward<Args>(idxs)...);

            // Wrap in the rank-appropriate base so fn can do algebra
            wrapped_fiber_type wrapped(std::in_place, std::move(fiber));

            // Apply the transform function to the wrapped fiber (by ref
            // since wrapped types inheriting NonReturnable are non-copyable)
            auto result = m_fn(std::as_const(wrapped));

            // Index into the result to get the single coefficient.
            // The fiber dimensions in the original index tuple tell us
            // which position within the result fiber to read.
            return _index_result(result, std::forward<Args>(idxs)...);
        }

        template <typename... Args>
        value_type coeff(Args &&...idxs) const {
            zipper::utils::extents::indices_in_range(this->extents(), idxs...);
            return _coeff(
                std::make_integer_sequence<rank_type,
                                           child_extents_type::rank()>{},
                std::forward<Args>(idxs)...);
        }

        // ── assign_to() — fast fiber-by-fiber path ─────────────────────────
        //
        // Called by AssignHelper when it detects HasCustomAssignStrategy.
        // Iterates over non-fiber index combinations, applies fn once per
        // fiber, writes the result to the target.

        template <zipper::concepts::Expression To>
        void assign_to(To &to) const {
            auto non_fiber_ext =
                index_remover::get_extents(expression().extents());
            auto full_seq =
                std::make_integer_sequence<rank_type,
                                           child_extents_type::rank()>{};

            // Iterate over non-fiber indices
            zipper::utils::extents::for_each_index_row_major(
                non_fiber_ext, [&](auto... non_fiber_idxs) {
                    // Build the fiber slice from the source expression
                    auto fiber =
                        _make_fiber_from_non_fiber(full_seq, non_fiber_idxs...);

                    // Wrap in base so fn can do algebra
                    wrapped_fiber_type wrapped(std::in_place, std::move(fiber));

                    // Apply transform (by ref: non-copyable due to
                    // NonReturnable)
                    auto result = m_fn(std::as_const(wrapped));

                    // Write result back to target along this fiber
                    _write_fiber_to_target(
                        to, result, full_seq, non_fiber_idxs...);
                });
        }

        /// Deep-copy child so the result owns all data.
        auto make_owned() const {
            auto owned_child = expression().make_owned();
            return PartialTransform<const decltype(owned_child),
                                    Fn,
                                    Indices...>(std::move(owned_child), m_fn);
        }

      private:
        Fn m_fn;

        // ── Helper: index into fn result using fiber coordinates ───────────

        /// Extract the fiber-dimension coordinates from a full index tuple
        /// and use them to index into the transformed result.
        template <typename Result, typename... Args>
        value_type _index_result(const Result &result,
                                 Args &&...full_idxs) const {
            if constexpr (sizeof...(Indices) == 1) {
                // Single fiber dimension: result is rank-1, index with
                // the coordinate at the fiber dimension.
                constexpr rank_type fiber_dim = (Indices, ...);
                auto fiber_idx =
                    zipper::detail::pack_index<fiber_dim>(full_idxs...);
                return result(fiber_idx);
            } else {
                // Multiple fiber dimensions: index with all fiber coords.
                return _index_result_multi(result,
                                           std::index_sequence<Indices...>{},
                                           std::forward<Args>(full_idxs)...);
            }
        }

        template <typename Result, rank_type... FiberDims, typename... Args>
        value_type
            _index_result_multi(const Result &result,
                                std::integer_sequence<rank_type, FiberDims...>,
                                Args &&...full_idxs) const {
            return result(
                zipper::detail::pack_index<FiberDims>(full_idxs...)...);
        }

        // ── Helper: build fiber slice from non-fiber indices ───────────────

        /// Given non-fiber index values, expand them into the full child
        /// index space (full_extent_t at fiber dims, index_type at others)
        /// and construct a Slice.
        template <rank_type... N, typename... NonFiberArgs>
        auto _make_fiber_from_non_fiber(std::integer_sequence<rank_type, N...>,
                                        NonFiberArgs... non_fiber_idxs) const {
            return slice_type(expression(),
                              _expand_non_fiber_index<N>(non_fiber_idxs...)...);
        }

        template <rank_type N, typename... NonFiberArgs>
        auto _expand_non_fiber_index(NonFiberArgs... non_fiber_idxs) const {
            if constexpr (((N == Indices) || ...)) {
                return zipper::full_extent_t{};
            } else {
                constexpr rank_type reduced_idx =
                    index_remover::full_rank_to_reduced_indices[N];
                return zipper::detail::pack_index<reduced_idx>(
                    non_fiber_idxs...);
            }
        }

        // ── Helper: write transformed fiber to target ──────────────────────

        /// Write the result of fn(fiber) into the target expression at the
        /// appropriate positions.
        ///
        /// Uses array-based index assembly to avoid the fundamental C++
        /// problem of splitting two trailing variadic parameter packs.
        template <typename To,
                  typename Result,
                  rank_type... N,
                  typename... NonFiberArgs>
        void _write_fiber_to_target(To &to,
                                    const Result &result,
                                    std::integer_sequence<rank_type, N...>,
                                    NonFiberArgs... non_fiber_idxs) const {
            constexpr rank_type R = child_extents_type::rank();

            // Build a partial index array with non-fiber indices placed at
            // their full-rank positions. Fiber positions will be filled
            // during iteration.
            std::array<index_type, R> base_idx{};
            _place_non_fiber_into(
                base_idx,
                std::make_index_sequence<sizeof...(NonFiberArgs)>{},
                non_fiber_idxs...);

            // Get the fiber extents for iteration
            auto fiber_ext =
                _get_fiber_extents(std::index_sequence<Indices...>{});

            // Iterate over all fiber index combinations
            zipper::utils::extents::for_each_index_row_major(
                fiber_ext, [&](auto... fiber_idxs) {
                    // Fill fiber positions into a copy of the base index
                    auto full_idx = base_idx;
                    _place_fiber_into(
                        full_idx,
                        std::make_index_sequence<sizeof...(Indices)>{},
                        std::integer_sequence<rank_type, Indices...>{},
                        fiber_idxs...);

                    // Index into fn's result with fiber indices only
                    auto result_val = result(fiber_idxs...);

                    // Write to target using the assembled full index
                    [&]<rank_type... Is>(
                        std::integer_sequence<rank_type, Is...>) {
                        to(full_idx[Is]...) = result_val;
                    }(std::make_integer_sequence<rank_type, R>{});
                });
        }

        template <rank_type... FiberDims>
        auto _get_fiber_extents(
            std::integer_sequence<rank_type, FiberDims...>) const {
            if constexpr (sizeof...(FiberDims) == 1) {
                constexpr rank_type dim = (FiberDims, ...);
                return zipper::extents<child_extents_type::static_extent(dim)>(
                    expression().extent(dim));
            } else {
                return zipper::extents<child_extents_type::static_extent(
                    FiberDims)...>(expression().extent(FiberDims)...);
            }
        }

        /// Place non-fiber indices into their full-rank positions.
        template <std::size_t... Is, typename... NonFiberArgs>
        static void _place_non_fiber_into(
            std::array<index_type, child_extents_type::rank()> &idx,
            std::index_sequence<Is...>,
            NonFiberArgs... non_fiber_idxs) {
            ((idx[index_remover::reduced_rank_to_full_indices[Is]] =
                  zipper::detail::pack_index<Is>(non_fiber_idxs...)),
             ...);
        }

        /// Place fiber indices into their full-rank positions.
        template <std::size_t... Is,
                  rank_type... FiberDims,
                  typename... FiberArgs>
        static void _place_fiber_into(
            std::array<index_type, child_extents_type::rank()> &idx,
            std::index_sequence<Is...>,
            std::integer_sequence<rank_type, FiberDims...>,
            FiberArgs... fiber_idxs) {
            ((idx[FiberDims] = zipper::detail::pack_index<Is>(fiber_idxs...)),
             ...);
        }
    };

} // namespace unary
} // namespace zipper::expression
#endif
