#include "catch_include.hpp"

#include "zipper/detail/declare_operations.hpp"
#include "zipper/views/binary/ArithmeticViews.hpp"
#include "zipper/views/nullary/ConstantView.hpp"
#include "zipper/views/unary/ScalarArithmeticViews.hpp"

namespace zipper {
template <concepts::ViewDerived View>
class TestBase {
   public:
    using value_type = View::value_type;
    using view_type = View;
    TestBase() = default;
    TestBase(View&& v) : m_view(v) {}

    View m_view;

    View& view() { return m_view; }
    const View& view() const { return m_view; }
};
namespace concepts {
namespace detail {
template <typename>
struct IsTestBase : std::false_type {};

template <typename T>
struct IsTestBase<TestBase<T>> : std::true_type {};
}  // namespace detail
template <typename T>
concept TestBaseDerived =
    std::derived_from<T, TestBase<T>> || detail::IsTestBase<T>::value;
;
}  // namespace concepts
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
    zipper::TestBase<zipper::views::nullary::ConstantView<double, 5>> A(5);
    -A;
    2 - (double(5) * A);
    zipper::TestBase<zipper::views::nullary::ConstantView<double, 5>> B(4);
    zipper::TestBase<zipper::views::nullary::ConstantView<bool, 5>> C(true);

    !(4.0 < A);
    A* B;
    (A > 4.0) == C;
    (A < 4.0) == C;
}
