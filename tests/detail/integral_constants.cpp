
#include "catch_include.hpp"
#include <zipper/detail/constexpr_arithmetic.hpp>

using namespace zipper;
using namespace zipper::detail;
TEST_CASE("test_constexpr_arithmetic", "[indices]") {
    std::integral_constant<index_type, 2> Two;
    std::integral_constant<index_type, 3> Three;

    index_type two = 2;
    index_type three = 3;

    {
        constexpr static int res = 5;
        auto a = plus(two, three);
        STATIC_CHECK(std::is_same_v<decltype(a), index_type>);
        CHECK(a == res);
        auto b = plus(two, Three);
        STATIC_CHECK(std::is_same_v<decltype(b), index_type>);
        CHECK(b == res);
        auto c = plus(Two, three);
        STATIC_CHECK(std::is_same_v<decltype(c), index_type>);
        CHECK(c == res);
        auto d = plus(Two, Three);
        STATIC_CHECK(std::is_same_v<decltype(d),
                                     std::integral_constant<index_type, res>>);
        CHECK(d == res);
    }

    {
        constexpr static index_type res = -1;
        auto a = minus(two, three);
        STATIC_CHECK(std::is_same_v<decltype(a), index_type>);
        CHECK(a == res);
        auto b = minus(two, Three);
        STATIC_CHECK(std::is_same_v<decltype(b), index_type>);
        CHECK(b == res);
        auto c = minus(Two, three);
        STATIC_CHECK(std::is_same_v<decltype(c), index_type>);
        CHECK(c == res);
        auto d = minus(Two, Three);
        STATIC_CHECK(std::is_same_v<decltype(d),
                                     std::integral_constant<index_type, res>>);
        CHECK(d == res);
    }
    {
        constexpr static index_type res = 6;
        auto a = multiplies(two, three);
        STATIC_CHECK(std::is_same_v<decltype(a), index_type>);
        CHECK(a == res);
        auto b = multiplies(two, Three);
        STATIC_CHECK(std::is_same_v<decltype(b), index_type>);
        CHECK(b == res);
        auto c = multiplies(Two, three);
        STATIC_CHECK(std::is_same_v<decltype(c), index_type>);
        CHECK(c == res);
        auto d = multiplies(Two, Three);
        STATIC_CHECK(std::is_same_v<decltype(d),
                                     std::integral_constant<index_type, res>>);
        CHECK(d == res);
    }
}
