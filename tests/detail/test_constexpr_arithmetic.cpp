
#include <zipper/detail/constexpr_arithmetic.hpp>
#include <zipper/types.hpp>

#include "../catch_include.hpp"

// runtime
using IR = zipper::index_type;
// compiletime
template <zipper::index_type T>
using IC = zipper::static_index_t<T>;

template <zipper::concepts::IndexLike T>
using CA = zipper::detail::ConstexprArithmetic<T>;

using namespace zipper::detail;
namespace {
template <typename BINO, IR A, IR B, IR R>
void check_binop() {
    {
        auto r = apply_binop<BINO, IC<A>, IC<B>>({}, {});
        using RT = std::decay_t<decltype(r)>;
        static_assert(std::is_same_v<RT, IC<R>>);
    }
    {
        auto r = apply_binop<BINO, IR, IC<B>>(A, {});
        using RT = std::decay_t<decltype(r)>;
        static_assert(std::is_same_v<RT, IR>);
        CHECK(r == R);
    }
    {
        auto r = apply_binop<BINO, IC<A>, IR>({}, B);
        using RT = std::decay_t<decltype(r)>;
        static_assert(std::is_same_v<RT, IR>);
        CHECK(r == R);
    }
    {
        auto r = apply_binop<BINO, IR, IR>(A, B);
        using RT = std::decay_t<decltype(r)>;
        static_assert(std::is_same_v<RT, IR>);
        CHECK(r == R);
    }
}
}  // namespace

TEST_CASE("test_constexpr_arithmetic_binops",
          "[detail][constexpr_arithmetic]") {
    check_binop<std::plus<IR>, 1, 2, 3>();
    check_binop<std::minus<IR>, 5, 2, 3>();
    check_binop<std::multiplies<IR>, 3, 2, 6>();
    check_binop<std::divides<IR>, 12, 3, 4>();
    check_binop<std::modulus<IR>, 12, 5, 2>();

    check_binop<std::plus<IR>, std::dynamic_extent, 2, std::dynamic_extent>();
    check_binop<std::minus<IR>, std::dynamic_extent, 2, std::dynamic_extent>();
    check_binop<std::multiplies<IR>, std::dynamic_extent, 2,
                std::dynamic_extent>();
    check_binop<std::divides<IR>, std::dynamic_extent, 2,
                std::dynamic_extent>();
    check_binop<std::modulus<IR>, std::dynamic_extent, 2,
                std::dynamic_extent>();
    check_binop<std::plus<IR>, 1, std::dynamic_extent, std::dynamic_extent>();
    check_binop<std::minus<IR>, 1, std::dynamic_extent, std::dynamic_extent>();
    check_binop<std::multiplies<IR>, 1, std::dynamic_extent,
                std::dynamic_extent>();
    check_binop<std::divides<IR>, 1, std::dynamic_extent,
                std::dynamic_extent>();
    check_binop<std::modulus<IR>, 1, std::dynamic_extent,
                std::dynamic_extent>();
}

TEST_CASE("test_constexpr_arithmetic_runtime",
          "[detail][constexpr_arithmetic]") {
#if defined(ZIPPER_DONT_USE_ALIAS_CTAD)
    CA<zipper::index_type> R3(3);
    CA<zipper::index_type> R4(4);
    CA<zipper::index_type> R6(6);
    CA<zipper::index_type> R12(12);
    CA<zipper::index_type> RD(std::dynamic_extent);
#else
    CA R3(3);
    CA R4(4);
    CA R6(6);
    CA R12(12);
    CA RD(std::dynamic_extent);
#endif

    static_assert(std::is_same_v<std::decay_t<decltype(R3)>, CA<IR>>);
    static_assert(std::is_same_v<std::decay_t<decltype(R4)>, CA<IR>>);
    static_assert(std::is_same_v<std::decay_t<decltype(R6)>, CA<IR>>);
    static_assert(std::is_same_v<std::decay_t<decltype(R12)>, CA<IR>>);

    auto R7 = R3 + R4;
    static_assert(std::is_same_v<std::decay_t<decltype(R7)>, CA<IR>>);
    CHECK(IR(R7) == 7);

    auto R2 = R6 - R4;
    static_assert(std::is_same_v<std::decay_t<decltype(R2)>, CA<IR>>);
    CHECK(IR(R2) == 2);

    auto R24 = R6 * R4;
    static_assert(std::is_same_v<std::decay_t<decltype(R24)>, CA<IR>>);
    CHECK(IR(R24) == 24);

    auto R8 = R24 / R3;
    static_assert(std::is_same_v<std::decay_t<decltype(R8)>, CA<IR>>);
    CHECK(IR(R8) == 8);

    auto R5 = R12 % R7;
    static_assert(std::is_same_v<std::decay_t<decltype(R5)>, CA<IR>>);
    CHECK(IR(R5) == 5);

    CHECK(IR(R3 + RD) == std::dynamic_extent);
    CHECK(IR(R3 - RD) == std::dynamic_extent);
    CHECK(IR(R3 * RD) == std::dynamic_extent);
    CHECK(IR(R3 / RD) == std::dynamic_extent);
    CHECK(IR(R3 % RD) == std::dynamic_extent);

    CHECK(IR(RD + R3) == std::dynamic_extent);
    CHECK(IR(RD - R3) == std::dynamic_extent);
    CHECK(IR(RD * R3) == std::dynamic_extent);
    CHECK(IR(RD / R3) == std::dynamic_extent);
    CHECK(IR(RD % R3) == std::dynamic_extent);
}
TEST_CASE("test_constexpr_arithmetic_compiletime",
          "[detail][constexpr_arithmetic]") {
#if defined(ZIPPER_DONT_USE_ALIAS_CTAD)
    CA<IC<3>> R3(IC<3>{});
    CA<IC<4>> R4(IC<4>{});
    CA<IC<6>> R6(IC<6>{});
    CA<IC<12>> R12(IC<12>{});
    CA<IC<std::dynamic_extent>> RD(IC<std::dynamic_extent>{});
#else
    CA R3(IC<3>{});
    CA R4(IC<4>{});
    CA R6(IC<6>{});
    CA R12(IC<12>{});
    CA RD(IC<std::dynamic_extent>{});
#endif

    static_assert(std::is_same_v<std::decay_t<decltype(R3)>, CA<IC<3>>>);
    static_assert(std::is_same_v<std::decay_t<decltype(R4)>, CA<IC<4>>>);
    static_assert(std::is_same_v<std::decay_t<decltype(R6)>, CA<IC<6>>>);
    static_assert(std::is_same_v<std::decay_t<decltype(R12)>, CA<IC<12>>>);

    auto R7 = R3 + R4;
    static_assert(std::is_same_v<std::decay_t<decltype(R7)>, CA<IC<7>>>);
    CHECK(IR(R7) == 7);

    auto R2 = R6 - R4;
    static_assert(std::is_same_v<std::decay_t<decltype(R2)>, CA<IC<2>>>);
    CHECK(IR(R2) == 2);

    auto R24 = R6 * R4;
    static_assert(std::is_same_v<std::decay_t<decltype(R24)>, CA<IC<24>>>);
    CHECK(IR(R24) == 24);

    auto R8 = R24 / R3;
    static_assert(std::is_same_v<std::decay_t<decltype(R8)>, CA<IC<8>>>);
    CHECK(IR(R8) == 8);

    auto R5 = R12 % R7;
    static_assert(std::is_same_v<std::decay_t<decltype(R5)>, CA<IC<5>>>);
    CHECK(IR(R5) == 5);
    CHECK(IR(R3 + RD) == std::dynamic_extent);
    CHECK(IR(R3 - RD) == std::dynamic_extent);
    CHECK(IR(R3 * RD) == std::dynamic_extent);
    CHECK(IR(R3 / RD) == std::dynamic_extent);
    CHECK(IR(R3 % RD) == std::dynamic_extent);

    CHECK(IR(RD + R3) == std::dynamic_extent);
    CHECK(IR(RD - R3) == std::dynamic_extent);
    CHECK(IR(RD * R3) == std::dynamic_extent);
    CHECK(IR(RD / R3) == std::dynamic_extent);
    CHECK(IR(RD % R3) == std::dynamic_extent);
}

TEST_CASE("test_constexpr_arithmetic_mix", "[detail][constexpr_arithmetic]") {
#if defined(ZIPPER_DONT_USE_ALIAS_CTAD)
    CA<zipper::index_type> R3(3);
    CA<zipper::index_type> R4(4);
    CA<zipper::index_type> R6(6);
    CA<zipper::index_type> R12(12);
    CA<zipper::index_type> RD(std::dynamic_extent);
    CA<IC<3>> C3(IC<3>{});
    CA<IC<4>> C4(IC<4>{});
    CA<IC<6>> C6(IC<6>{});
    CA<IC<12>> C12(IC<12>{});
    CA<IC<std::dynamic_extent>> CD(IC<std::dynamic_extent>{});
#else
    CA R3(3);
    CA R4(4);
    CA R6(6);
    CA R12(12);
    CA RD(std::dynamic_extent);
    CA C3(IC<3>{});
    CA C4(IC<4>{});
    CA C6(IC<6>{});
    CA C12(IC<12>{});
    CA CD(IC<std::dynamic_extent>{});
#endif

    {
        auto R7 = C3 + R4;
        static_assert(std::is_same_v<std::decay_t<decltype(R7)>, CA<IR>>);
        CHECK(IR(R7) == 7);

        auto R2 = C6 - R4;
        static_assert(std::is_same_v<std::decay_t<decltype(R2)>, CA<IR>>);
        CHECK(IR(R2) == 2);

        auto R24 = C6 * R4;
        static_assert(std::is_same_v<std::decay_t<decltype(R24)>, CA<IR>>);
        CHECK(IR(R24) == 24);

#if defined(ZIPPER_DONT_USE_ALIAS_CTAD)
        CA<IC<24> C24(IC<24>{});
#else
        CA C24(IC<24>{});
#endif
        auto R8 = C24 / R3;
        static_assert(std::is_same_v<std::decay_t<decltype(R8)>, CA<IR>>);
        CHECK(IR(R8) == 8);

        auto R5 = C12 % R7;
        static_assert(std::is_same_v<std::decay_t<decltype(R5)>, CA<IR>>);
        CHECK(IR(R5) == 5);
    }

    {
        auto R7 = R3 + C4;
        static_assert(std::is_same_v<std::decay_t<decltype(R7)>, CA<IR>>);
        CHECK(IR(R7) == 7);

        auto R2 = R6 - C4;
        static_assert(std::is_same_v<std::decay_t<decltype(R2)>, CA<IR>>);
        CHECK(IR(R2) == 2);

        auto R24 = R6 * C4;
        static_assert(std::is_same_v<std::decay_t<decltype(R24)>, CA<IR>>);
        CHECK(IR(R24) == 24);

        auto R8 = R24 / C3;
        static_assert(std::is_same_v<std::decay_t<decltype(R8)>, CA<IR>>);
        CHECK(IR(R8) == 8);

#if defined(ZIPPER_DONT_USE_ALIAS_CTAD)
        CA<IC<7> C24(IC<7>{});
#else
        CA C7(IC<7>{});
#endif
        auto R5 = R12 % C7;
        static_assert(std::is_same_v<std::decay_t<decltype(R5)>, CA<IR>>);
        CHECK(IR(R5) == 5);
    }

    CHECK(IR(C3 + RD) == std::dynamic_extent);
    CHECK(IR(C3 - RD) == std::dynamic_extent);
    CHECK(IR(C3 * RD) == std::dynamic_extent);
    CHECK(IR(C3 / RD) == std::dynamic_extent);
    CHECK(IR(C3 % RD) == std::dynamic_extent);

    CHECK(IR(CD + R3) == std::dynamic_extent);
    CHECK(IR(CD - R3) == std::dynamic_extent);
    CHECK(IR(CD * R3) == std::dynamic_extent);
    CHECK(IR(CD / R3) == std::dynamic_extent);
    CHECK(IR(CD % R3) == std::dynamic_extent);

    CHECK(IR(R3 + CD) == std::dynamic_extent);
    CHECK(IR(R3 - CD) == std::dynamic_extent);
    CHECK(IR(R3 * CD) == std::dynamic_extent);
    CHECK(IR(R3 / CD) == std::dynamic_extent);
    CHECK(IR(R3 % CD) == std::dynamic_extent);

    CHECK(IR(RD + C3) == std::dynamic_extent);
    CHECK(IR(RD - C3) == std::dynamic_extent);
    CHECK(IR(RD * C3) == std::dynamic_extent);
    CHECK(IR(RD / C3) == std::dynamic_extent);
    CHECK(IR(RD % C3) == std::dynamic_extent);
}
