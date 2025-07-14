
#include "catch_include.hpp"
#include <zipper/concepts/shapes.hpp>
#include <zipper/detail/extents/dynamic_extents_indices.hpp>
#include <zipper/detail/extents/is_compatible.hpp>
#include <zipper/detail/extents/swizzle_extents.hpp>
#include <zipper/storage/PlainObjectStorage.hpp>
#include <zipper/types.hpp>
#include <zipper/views/unary/detail/invert_integer_sequence.hpp>

using namespace zipper;
TEST_CASE("test_extent_compatibility", "[extents]") {
    // detail::extents::is_compatible(extents<3,3>{}, extents<3,3>{});
    CHECK(detail::extents::is_compatible<3, 3>(extents<3, 3>{}));
    CHECK(detail::extents::is_compatible<3>(extents<3>{}));
    CHECK(detail::extents::is_compatible<3, 2>(
        extents<3, std::dynamic_extent>{2}));
    CHECK(detail::extents::is_compatible<2>(extents<std::dynamic_extent>{2}));
    CHECK_FALSE(detail::extents::is_compatible<3, 3>(extents<2, 3>{}));
    CHECK_FALSE(detail::extents::is_compatible<3>(extents<4>{}));
    CHECK_FALSE(detail::extents::is_compatible<3, 2>(
        extents<2, std::dynamic_extent>{3}));
    CHECK_FALSE(detail::extents::is_compatible<3, 2>(
        extents<3, std::dynamic_extent>{3}));
    CHECK_FALSE(
        detail::extents::is_compatible<2>(extents<std::dynamic_extent>{3}));

    CHECK(zipper::concepts::ValidExtents<extents<3, 3>, 3, 3>);
    CHECK(
        zipper::concepts::ValidExtents<extents<std::dynamic_extent, 3>, 3, 3>);
    CHECK(
        zipper::concepts::ValidExtents<extents<3, std::dynamic_extent>, 3, 3>);
    CHECK(
        zipper::concepts::ValidExtents<extents<3, 3>, 3, std::dynamic_extent>);
    CHECK(zipper::concepts::ValidExtents<extents<std::dynamic_extent, 3>, 3,
                                         std::dynamic_extent>);
    CHECK(zipper::concepts::ValidExtents<extents<3, std::dynamic_extent>, 3,
                                         std::dynamic_extent>);
}
TEST_CASE("test_invert_integer_sequence", "[extents]") {
    auto to_array = []<rank_type... N>(std::integer_sequence<rank_type, N...>) {
        return std::array<rank_type, sizeof...(N)>{{N...}};
    };
    constexpr static std::array<index_type, 3> A{{0, 1, 2}};
    {
        auto a = to_array(std::integer_sequence<rank_type>{});
        auto b = to_array(
            zipper::views::unary::detail::invert_integer_sequence<3, 0, 1,
                                                                  2>::type{});

        static_assert(a.size() == b.size());
        CHECK(a == b);

        using E = zipper::views::unary::detail::invert_integer_sequence<
            3, 0, 1, 2>::assign_types<zipper::extents, A>;
        static_assert(std::is_same_v<E, zipper::extents<>>);
    }
    {
        auto a = to_array(std::integer_sequence<rank_type, 0>{});
        auto b = to_array(
            zipper::views::unary::detail::invert_integer_sequence<3, 1,
                                                                  2>::type{});

        static_assert(a.size() == b.size());
        CHECK(a == b);

        using E = zipper::views::unary::detail::invert_integer_sequence<
            3, 1, 2>::assign_types<zipper::extents, A>;
        static_assert(std::is_same_v<E, zipper::extents<0>>);
    }
    {
        auto a = to_array(std::integer_sequence<rank_type, 0, 2>{});
        auto b = to_array(
            zipper::views::unary::detail::invert_integer_sequence<3,
                                                                  1>::type{});

        static_assert(a.size() == b.size());
        CHECK(a == b);
    }
    {
        auto a = to_array(std::integer_sequence<rank_type, 0, 1, 2>{});
        auto b = to_array(
            zipper::views::unary::detail::invert_integer_sequence<3>::type{});

        static_assert(a.size() == b.size());
        CHECK(a == b);

        using E = zipper::views::unary::detail::invert_integer_sequence<
            3>::assign_types<zipper::extents, A>;
        static_assert(std::is_same_v<E, zipper::extents<0, 1, 2>>);
    }
}

TEST_CASE("test_extents_dynamic_size", "[extents][dense]") {
    {
        zipper::extents<5, std::dynamic_extent> E(19);
        constexpr static auto dyn =
            zipper::detail::extents::dynamic_extents_indices_v<decltype(E)>;
        static_assert(dyn.size() == 1);
        static_assert(dyn[0] == 1);
        auto val = zipper::detail::extents::dynamic_extents(E);
        static_assert(val.size() == 1);
        CHECK(val[0] == 19);
    }
    {
        zipper::extents<std::dynamic_extent, 5, std::dynamic_extent> E(3, 5);
        constexpr static auto dyn =
            zipper::detail::extents::dynamic_extents_indices_v<decltype(E)>;
        static_assert(dyn.size() == 2);
        static_assert(dyn[0] == 0);
        static_assert(dyn[1] == 2);
        auto val = zipper::detail::extents::dynamic_extents(E);
        static_assert(val.size() == 2);
        CHECK(val[0] == 3);
        CHECK(val[1] == 5);
    }
}
TEST_CASE("test_extents_swizzle", "[extents][dense]") {
    {
        zipper::extents<5, std::dynamic_extent> E(19);
        auto SE = zipper::detail::extents::swizzle_extents(
            E, std::integer_sequence<zipper::index_type, 1, 0>{});

        REQUIRE(SE.rank() == 2);
        CHECK(SE.static_extent(0) == std::dynamic_extent);
        CHECK(SE.static_extent(1) == 5);
        CHECK(SE == zipper::dextents<2>(19, 5));
    }

    {
        zipper::extents<std::dynamic_extent, 5, std::dynamic_extent> E(3, 6);
        REQUIRE(E == zipper::dextents<3>(3, 5, 6));

        auto e3 = zipper::detail::extents::swizzle_extents(
            E, std::integer_sequence<zipper::index_type, 0>{});
        static_assert(decltype(e3)::rank() == 1);
        static_assert(decltype(e3)::static_extent(0) == std::dynamic_extent);
        CHECK(e3.extent(0) == 3);

        CHECK(zipper::detail::extents::swizzle_extents(
                  E, std::integer_sequence<zipper::index_type, 0, 1>{}) ==
              zipper::create_dextents(3, 5));
        CHECK(zipper::detail::extents::swizzle_extents(
                  E, std::integer_sequence<zipper::index_type, 1, 0>{}) ==
              zipper::create_dextents(5, 3));

        // CHECK(zipper::detail::extents::swizzle_extents(
        //           E, std::integer_sequence<zipper::index_type, 1, 1>{}) ==
        //       zipper::create_dextents(5, 5));

        CHECK(zipper::detail::extents::swizzle_extents(
                  E, std::integer_sequence<zipper::index_type, 2, 1, 0>{}) ==
              zipper::create_dextents(6, 5, 3));

        // zipper::detail::ExtentsSwizzler<2, 1, 0, 1> s;
        // auto se = s.swizzle_extents(E);
        // CHECK(se == zipper::create_dextents(6, 5, 3, 5));

        // auto arr = s.swizzle(10, 11, 12);
        // REQUIRE(arr.size() == 4);
        // auto arr2 = s.swizzle(std::make_tuple(10, 11, 12));
        // REQUIRE(arr2.size() == 4);
        // CHECK(arr == arr2);
        // CHECK(arr[0] == 12);
        // CHECK(arr[1] == 11);
        // CHECK(arr[2] == 10);
        // CHECK(arr[3] == 11);
    }
}
