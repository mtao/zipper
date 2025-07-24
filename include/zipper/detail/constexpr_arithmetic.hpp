
#include <cstdlib>
#include <memory>
#if !defined(ZIPPER_DETAIL_CONSTEXPR_ARITHMETIC_HPP)
#define ZIPPER_DETAIL_CONSTEXPR_ARITHMETIC_HPP

#include <algorithm>  // for min / max

#include "zipper/concepts/IndexLike.hpp"
#include "zipper/types.hpp"

namespace zipper::detail {

template <typename BinOp, concepts::IndexLike A, concepts::IndexLike B>
constexpr auto apply_binop(const A& a, const B& b) {
    if constexpr (std::is_integral_v<A> || std::is_integral_v<B>) {
        if (a == std::dynamic_extent || b == std::dynamic_extent) {
            return std::dynamic_extent;
        }
        return BinOp{}(a, b);
    } else {
        if constexpr (A::value == std::dynamic_extent ||
                      B::value == std::dynamic_extent) {
            return std::integral_constant<index_type, std::dynamic_extent>{};
        } else {
            return std::integral_constant<index_type,
                                          BinOp{}(A::value, B::value)>{};
        }
    }
}

template <concepts::IndexLike A, concepts::IndexLike B>
constexpr auto plus(const A& a, const B& b) {
    return apply_binop<std::plus<index_type>, A, B>(a, b);
}
template <concepts::IndexLike A, concepts::IndexLike B>
constexpr auto minus(const A& a, const B& b) {
    return apply_binop<std::minus<index_type>, A, B>(a, b);
}
template <concepts::IndexLike A, concepts::IndexLike B>
constexpr auto multiplies(const A& a, const B& b) {
    return apply_binop<std::multiplies<index_type>, A, B>(a, b);
}
template <concepts::IndexLike A, concepts::IndexLike B>
constexpr auto divides(const A& a, const B& b) {
    return apply_binop<std::divides<index_type>, A, B>(a, b);
}
template <concepts::IndexLike A, concepts::IndexLike B>
constexpr auto modulus(const A& a, const B& b) {
    return apply_binop<std::modulus<index_type>, A, B>(a, b);
}
template <concepts::IndexLike Type>
struct ConstexprArithmetic {
    constexpr ConstexprArithmetic(const Type& t = {}) : m_type(t) {}
    constexpr bool is_dynamic() const { return m_type == std::dynamic_extent; }
    consteval static auto is_runtime() -> bool {
        return std::is_same_v<Type, index_type>;
    }
    consteval static auto is_compiletime() -> bool { return !is_runtime(); }

    constexpr operator index_type() const { return index_type(m_type); }

    Type m_type;
};

template <concepts::IndexLike Type>
ConstexprArithmetic(const Type& t) -> ConstexprArithmetic<
    std::conditional_t<std::is_integral_v<Type>, index_type, Type>>;

template <concepts::IndexLike Type, concepts::IndexLike Type2>
auto operator+(const ConstexprArithmetic<Type>& a,
               const ConstexprArithmetic<Type2>& b) {
    auto v = plus(a.m_type, b.m_type);
    return ConstexprArithmetic(v);
}
template <concepts::IndexLike Type, concepts::IndexLike Type2>
auto operator-(const ConstexprArithmetic<Type>& a,
               const ConstexprArithmetic<Type2>& b) {
    auto v = minus(a.m_type, b.m_type);
    return ConstexprArithmetic(v);
}
template <concepts::IndexLike Type, concepts::IndexLike Type2>
auto operator*(const ConstexprArithmetic<Type>& a,
               const ConstexprArithmetic<Type2>& b) {
    auto v = multiplies(a.m_type, b.m_type);
    return ConstexprArithmetic(v);
}
template <concepts::IndexLike Type, concepts::IndexLike Type2>
auto operator/(const ConstexprArithmetic<Type>& a,
               const ConstexprArithmetic<Type2>& b) {
    auto v = divides(a.m_type, b.m_type);
    return ConstexprArithmetic(v);
}
template <concepts::IndexLike Type, concepts::IndexLike Type2>
auto operator%(const ConstexprArithmetic<Type>& a,
               const ConstexprArithmetic<Type2>& b) {
    auto v = modulus(a.m_type, b.m_type);
    return ConstexprArithmetic(v);
}
}  // namespace zipper::detail
#endif
