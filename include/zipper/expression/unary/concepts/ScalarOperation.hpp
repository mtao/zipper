
#if !defined(ZIPPER_expression_UNARY_CONCEPTS_SCALAROPERATION_HPP)
#define ZIPPER_expression_UNARY_CONCEPTS_SCALAROPERATION_HPP
namespace zipper::expression::unary::concepts {

template <typename value_type, typename OpType>
concept ScalarOperation = requires(value_type v) { OpType{}(v); };
// template <typename value_type, template <typename> typename OpType>
// concept ScalarOperation = requires(value_type v) {
//     OpType<value_type>{}(v) -> value_type;
// };
} // namespace zipper::expression::unary::concepts
#endif
