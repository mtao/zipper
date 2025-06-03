
#if !defined(ZIPPER_DETAIL_CONSTEXPR_ARITHMETIC_HPP)
#define ZIPPER_DETAIL_CONSTEXPR_ARITHMETIC_HPP

#include "zipper/concepts/IndexLike.hpp"
#include "zipper/types.hpp"

namespace zipper::detail {

template <typename BinOp, concepts::IndexLike A, concepts::IndexLike B>
auto apply_binop(const A& a, const B& b) {
    if constexpr (std::is_integral_v<A> || std::is_integral_v<B>) {
        return BinOp{}(a, b);
    } else {
        return std::integral_constant<index_type,
                                      BinOp{}(A::value, B::value)>{};
    }
}

template <concepts::IndexLike A, concepts::IndexLike B>
auto plus(const A& a, const B& b) {
    return apply_binop<std::plus<index_type>, A, B>(a, b);
}
template <concepts::IndexLike A, concepts::IndexLike B>
auto minus(const A& a, const B& b) {
    return apply_binop<std::minus<index_type>, A, B>(a, b);
}
template <concepts::IndexLike A, concepts::IndexLike B>
auto multiplies(const A& a, const B& b) {
    return apply_binop<std::multiplies<index_type>, A, B>(a, b);
}
template <concepts::IndexLike A, concepts::IndexLike B>
auto divides(const A& a, const B& b) {
    return apply_binop<std::divides<index_type>, A, B>(a, b);
}
template <concepts::IndexLike A, concepts::IndexLike B>
auto modulus(const A& a, const B& b) {
    return apply_binop<std::modulus<index_type>, A, B>(a, b);
}

template <concepts::IndexLike A, concepts::IndexLike B>
auto min(const A& a, const B& b) {
    return apply_binop<std::min<index_type>, A, B>(a, b);
}

template <concepts::IndexLike A, concepts::IndexLike B>
auto max(const A& a, const B& b) {
    return apply_binop<std::max<index_type>, A, B>(a, b);
}

}  // namespace zipper::detail
#endif
