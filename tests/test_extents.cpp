
#include <catch2/catch_all.hpp>
#include <uvl/detail/extents/dynamic_extents_indices.hpp>
#include <uvl/detail/extents/swizzle_extents.hpp>
#include <uvl/storage/PlainObjectStorage.hpp>
#include <uvl/types.hpp>
#include <uvl/views/unary/detail/invert_integer_sequence.hpp>

using namespace uvl;
TEST_CASE("test_invert_integer_sequence", "[extents]") {
    auto to_array = []<rank_type... N>(std::integer_sequence<rank_type, N...>) {
        return std::array<rank_type, sizeof...(N)>{{N...}};
    };
    {
        auto a = to_array(std::integer_sequence<rank_type>{});
        auto b = to_array(
            uvl::views::unary::detail::invert_integer_sequence<3, 0, 1,
                                                               2>::type{});

        static_assert(a.size() == b.size());
        CHECK(a == b);

        using E = uvl::views::unary::detail::invert_integer_sequence<
            3, 0, 1, 2>::assign_types<uvl::extents>;
        static_assert(std::is_same_v<E, uvl::extents<>>);
    }
    {
        auto a = to_array(std::integer_sequence<rank_type, 0>{});
        auto b = to_array(
            uvl::views::unary::detail::invert_integer_sequence<3, 1,
                                                               2>::type{});

        static_assert(a.size() == b.size());
        CHECK(a == b);

        using E = uvl::views::unary::detail::invert_integer_sequence<
            3, 1, 2>::assign_types<uvl::extents>;
        static_assert(std::is_same_v<E, uvl::extents<0>>);
    }
    {
        auto a = to_array(std::integer_sequence<rank_type, 0, 2>{});
        auto b = to_array(
            uvl::views::unary::detail::invert_integer_sequence<3, 1>::type{});

        static_assert(a.size() == b.size());
        CHECK(a == b);
    }
    {
        auto a = to_array(std::integer_sequence<rank_type, 0, 1, 2>{});
        auto b = to_array(
            uvl::views::unary::detail::invert_integer_sequence<3>::type{});

        static_assert(a.size() == b.size());
        CHECK(a == b);

        using E = uvl::views::unary::detail::invert_integer_sequence<
            3>::assign_types<uvl::extents>;
        static_assert(std::is_same_v<E, uvl::extents<0, 1, 2>>);
    }
}

TEST_CASE("test_extents_dynamic_size", "[extents][dense]") {
    {
        uvl::extents<5, std::dynamic_extent> E(19);
        constexpr static auto dyn =
            uvl::detail::extents::dynamic_extents_indices_v<decltype(E)>;
        static_assert(dyn.size() == 1);
        static_assert(dyn[0] == 1);
        auto val = uvl::detail::extents::dynamic_extents(E);
        static_assert(val.size() == 1);
        CHECK(val[0] == 19);
    }
    {
        uvl::extents<std::dynamic_extent, 5, std::dynamic_extent> E(3, 5);
        constexpr static auto dyn =
            uvl::detail::extents::dynamic_extents_indices_v<decltype(E)>;
        static_assert(dyn.size() == 2);
        static_assert(dyn[0] == 0);
        static_assert(dyn[1] == 2);
        auto val = uvl::detail::extents::dynamic_extents(E);
        static_assert(val.size() == 2);
        CHECK(val[0] == 3);
        CHECK(val[1] == 5);
    }
}
TEST_CASE("test_extents_swizzle", "[extents][dense]") {
    {
        uvl::extents<5, std::dynamic_extent> E(19);
        auto SE = uvl::detail::extents::swizzle_extents(
            E, std::integer_sequence<uvl::index_type, 1, 0>{});

        REQUIRE(SE.rank() == 2);
        CHECK(SE.static_extent(0) == std::dynamic_extent);
        CHECK(SE.static_extent(1) == 5);
        CHECK(SE == uvl::dextents<2>(19, 5));
    }

    {
        uvl::extents<std::dynamic_extent, 5, std::dynamic_extent> E(3, 6);
        REQUIRE(E == uvl::dextents<3>(3, 5, 6));

        auto e3 = uvl::detail::extents::swizzle_extents(
            E, std::integer_sequence<uvl::index_type, 0>{});
        static_assert(decltype(e3)::rank() == 1);
        static_assert(decltype(e3)::static_extent(0) == std::dynamic_extent);
        CHECK(e3.extent(0) == 3);

        CHECK(uvl::detail::extents::swizzle_extents(
                  E, std::integer_sequence<uvl::index_type, 0, 1>{}) ==
              uvl::create_dextents(3, 5));
        CHECK(uvl::detail::extents::swizzle_extents(
                  E, std::integer_sequence<uvl::index_type, 1, 0>{}) ==
              uvl::create_dextents(5, 3));

        // CHECK(uvl::detail::extents::swizzle_extents(
        //           E, std::integer_sequence<uvl::index_type, 1, 1>{}) ==
        //       uvl::create_dextents(5, 5));

        CHECK(uvl::detail::extents::swizzle_extents(
                  E, std::integer_sequence<uvl::index_type, 2, 1, 0>{}) ==
              uvl::create_dextents(6, 5, 3));

        // uvl::detail::ExtentsSwizzler<2, 1, 0, 1> s;
        // auto se = s.swizzle_extents(E);
        // CHECK(se == uvl::create_dextents(6, 5, 3, 5));

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
