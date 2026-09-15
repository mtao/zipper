
#if !defined(ZIPPER_expression_UNARY_CONCEPTS_SCALAROPERATION_HPP)
#define ZIPPER_expression_UNARY_CONCEPTS_SCALAROPERATION_HPP
#include <type_traits>

namespace zipper::expression::unary::concepts {

// Structural callability only: const invocation does not imply purity.
template <typename value_type, typename OpType, typename... Other>
concept ScalarOperation = requires(const OpType &op, const value_type &v,
                                   const Other &...other) {
    requires(!std::is_void_v<decltype(op(v, other...))>);
};
} // namespace zipper::expression::unary::concepts
#endif
