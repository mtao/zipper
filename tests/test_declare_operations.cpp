#include "catch_include.hpp"

#include "zipper/concepts/Zipper.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"
#include "zipper/expression/nullary/Constant.hpp"
#include "zipper/expression/unary/ScalarArithmetic.hpp"
#include "zipper/expression/reductions/All.hpp"

namespace zipper {
template <concepts::QualifiedExpression View>
class TestBase {
   public:
    using expression_type = std::decay_t<View>;
    using value_type = typename expression::detail::ExpressionTraits<expression_type>::value_type;
    TestBase() = default;
    TestBase(View&& v) : m_expression(v) {}

    View m_expression;

    View& expression() { return m_expression; }
    const View& expression() const { return m_expression; }
};
namespace concepts::detail {

// Make TestBase satisfy concepts::Zipper via IsZipperBase
template <typename T>
struct IsZipperBase<TestBase<T>> : std::true_type {};

}  // namespace concepts::detail
UNARY_DECLARATION(TestBase, LogicalNot, operator!)
UNARY_DECLARATION(TestBase, BitNot, operator~)
UNARY_DECLARATION(TestBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(TestBase, Plus, operator+)
SCALAR_BINARY_DECLARATION(TestBase, Minus, operator-)
SCALAR_BINARY_DECLARATION(TestBase, Multiplies, operator*)
SCALAR_BINARY_DECLARATION(TestBase, Divides, operator/)
SCALAR_BINARY_DECLARATION(TestBase, Modulus, operator%)
SCALAR_BINARY_DECLARATION(TestBase, EqualsTo, operator==)
SCALAR_BINARY_DECLARATION(TestBase, NotEqualsTo, operator!=)
SCALAR_BINARY_DECLARATION(TestBase, Greater, operator>)
SCALAR_BINARY_DECLARATION(TestBase, Less, operator<)
SCALAR_BINARY_DECLARATION(TestBase, GreaterEqual, operator>=)
SCALAR_BINARY_DECLARATION(TestBase, LessEqual, operator<=)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
SCALAR_BINARY_DECLARATION(TestBase, LogicalAnd, operator&&)
SCALAR_BINARY_DECLARATION(TestBase, LogicalOr, operator||)
#pragma GCC diagnostic pop
SCALAR_BINARY_DECLARATION(TestBase, BitAnd, operator&)
SCALAR_BINARY_DECLARATION(TestBase, BitOr, operator|)
SCALAR_BINARY_DECLARATION(TestBase, BitXor, operator^)

BINARY_DECLARATION(TestBase, Plus, operator+)
BINARY_DECLARATION(TestBase, Minus, operator-)
BINARY_DECLARATION(TestBase, Multiplies, operator*)
BINARY_DECLARATION(TestBase, Divides, operator/)
BINARY_DECLARATION(TestBase, Modulus, operator%)
BINARY_DECLARATION(TestBase, EqualsTo, operator==)
BINARY_DECLARATION(TestBase, NotEqualsTo, operator!=)
BINARY_DECLARATION(TestBase, Greater, operator>)
BINARY_DECLARATION(TestBase, Less, operator<)
BINARY_DECLARATION(TestBase, GreaterEqual, operator>=)
BINARY_DECLARATION(TestBase, LessEqual, operator<=)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
BINARY_DECLARATION(TestBase, LogicalAnd, operator&&)
BINARY_DECLARATION(TestBase, LogicalOr, operator||)
#pragma GCC diagnostic pop
BINARY_DECLARATION(TestBase, BitAnd, operator&)
BINARY_DECLARATION(TestBase, BitOr, operator|)
BINARY_DECLARATION(TestBase, BitXor, operator^)

}  // namespace zipper

TEST_CASE("TestOpMacros", "[macros]") {
    zipper::TestBase<zipper::expression::nullary::Constant<double, 5>> A(5);
    -A;
    2 - (double(5) * A);
    zipper::TestBase<zipper::expression::nullary::Constant<double, 5>> B(4);
    zipper::TestBase<zipper::expression::nullary::Constant<bool, 5>> C(true);

    CHECK(zipper::expression::reductions::All(((A * B) == 20.0).expression())());
    CHECK(zipper::expression::reductions::All(((A > 4.0) == C).expression())());
    CHECK(zipper::expression::reductions::All(((A >= 5.0) == C).expression())());
    CHECK(zipper::expression::reductions::All((!((A < 4.0) == C)).expression())());
}
