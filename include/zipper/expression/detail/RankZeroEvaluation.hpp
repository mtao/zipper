#pragma once

#include "zipper/expression/binary/Operation.hpp"
#include "zipper/expression/unary/CoefficientWiseOperation.hpp"
#include "zipper/expression/unary/ScalarOperation.hpp"

namespace zipper::expression::nullary {
template <typename T, typename E, typename L, typename A> class MDArray;
template <typename T, typename E, typename L, typename A> class MDSpan;
template <typename T, index_type... N> class Constant;
template <typename T, int Value, index_type... N> class StaticConstant;
}

namespace zipper::expression::unary {
template <zipper::concepts::QualifiedExpression Child, typename... Slices>
class Slice;
}

namespace zipper::expression::detail {

// Evaluation domain only, never an alias-safety proof. For rank-zero sources,
// scalar values are evaluated once and broadcast, including opaque callbacks
// over scalar operands. Unknown index-callable expressions retain per-coordinate
// evaluation (notably Identity and Random, including Random's inherited coeff).
// Test coeff, not operator(): the latter has an unconstrained forwarding body.
template <typename From, rank_type DestinationRank>
struct RankZeroEvaluation {
    static constexpr bool is_scalar = []<std::size_t... I>(
        std::index_sequence<I...>) {
        return !requires(const From &from) {
            from.coeff((static_cast<void>(I), index_type{})...);
        };
    }(std::make_index_sequence<DestinationRank>{});
};

// These nodes ignore indices even though their coeff signatures are variadic.
template <typename T, typename E, typename L, typename A, rank_type R>
struct RankZeroEvaluation<nullary::MDArray<T, E, L, A>, R> {
    static constexpr bool is_scalar = true;
};
template <typename T, typename E, typename L, typename A, rank_type R>
struct RankZeroEvaluation<nullary::MDSpan<T, E, L, A>, R> {
    static constexpr bool is_scalar = true;
};
template <typename T, index_type... N, rank_type R>
struct RankZeroEvaluation<nullary::Constant<T, N...>, R> {
    static constexpr bool is_scalar = true;
};
template <typename T, int Value, index_type... N, rank_type R>
struct RankZeroEvaluation<nullary::StaticConstant<T, Value, N...>, R> {
    static constexpr bool is_scalar = true;
};
template <typename Child, typename... Slices, rank_type R>
struct RankZeroEvaluation<unary::Slice<Child, Slices...>, R> {
    static constexpr bool is_scalar = true;
};

template <typename Child, typename Op, rank_type R>
struct RankZeroEvaluation<unary::CoefficientWiseOperation<Child, Op>, R>
    : RankZeroEvaluation<std::remove_cvref_t<Child>, R> {};
template <typename Child, typename Op, typename Scalar, bool Right, rank_type R>
struct RankZeroEvaluation<unary::ScalarOperation<Child, Op, Scalar, Right>, R>
    : RankZeroEvaluation<std::remove_cvref_t<Child>, R> {};
template <typename Left, typename Right, typename Op, rank_type R>
struct RankZeroEvaluation<binary::Operation<Left, Right, Op>, R> {
    static constexpr bool is_scalar =
        RankZeroEvaluation<std::remove_cvref_t<Left>, R>::is_scalar
        && RankZeroEvaluation<std::remove_cvref_t<Right>, R>::is_scalar;
};

} // namespace zipper::expression::detail
