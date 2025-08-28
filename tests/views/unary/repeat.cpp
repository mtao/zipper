
#include <zipper/Array.hpp>
#include <zipper/views/nullary/RandomView.hpp>
#include <zipper/views/unary/RepeatView.hpp>

#include "../../catch_include.hpp"
#include "../../fmt_include.hpp"
using namespace zipper;
TEST_CASE("test_repeat_view", "[views][unary]") {
    {

        // This depends on P1814R0 ( CTAD for aggregates and aliases )
//#if defined(__cpp_deduction_guides) && __cplusplus >= 201907L
#if defined(ZIPPER_DONT_USE_ALIAS_CTAD)
        Array_ a = views::nullary::uniform_random_view<double>(extents<5>{});
#else
        Array a = views::nullary::uniform_random_view<double>(extents<5>{});
#endif
        REQUIRE(a.extents() == extents<5>{});

        auto l =
            views::unary::RepeatView<zipper::views::unary::RepeatMode::Left, 1,
                                     typename decltype(a)::view_type>(a.view());
        auto r =
            views::unary::RepeatView<zipper::views::unary::RepeatMode::Right, 1,
                                     typename decltype(a)::view_type>(a.view());
        for (index_type j = 0; j < a.extent(0); ++j) {
            for (index_type k = 0; k < 100; ++k) {
                CHECK(l(k, j) == a(j));
                CHECK(r(j, k) == a(j));
            }
        }
    }
    {
#if defined(ZIPPER_DONT_USE_ALIAS_CTAD)
        Array_ a = views::nullary::uniform_random_view<double>(extents<5, 3>{});
#else
        Array_ a = views::nullary::uniform_random_view<double>(extents<5, 3>{});
#endif
        REQUIRE(a.extents() == extents<5, 3>{});

        auto l =
            views::unary::RepeatView<zipper::views::unary::RepeatMode::Left, 2,
                                     typename decltype(a)::view_type>(a.view());
        auto r =
            views::unary::RepeatView<zipper::views::unary::RepeatMode::Right, 2,
                                     typename decltype(a)::view_type>(a.view());
        for (index_type j = 0; j < a.extent(0); ++j) {
            for (index_type k = 0; k < a.extent(1); ++k) {
                for (index_type jj = 0; jj < 100; ++jj) {
                    for (index_type kk = 0; kk < 100; ++kk) {
                        CHECK(l(jj, kk, j, k) == a(j, k));
                        CHECK(r(j, k, jj, kk) == a(j, k));
                    }
                }
            }
        }
    }
}
