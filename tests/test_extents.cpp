
#include <catch2/catch_all.hpp>
#include <uvl/detail/dynamic_extents_indices.hpp>
#include <uvl/detail/swizzle_extents.hpp>
#include <uvl/storage/PlainObjectStorage.hpp>
#include <uvl/types.hpp>

TEST_CASE("test_extents_dynamic_size", "[extents][dense]") {
    {
        uvl::extents<5, std::dynamic_extent> E(19);
        constexpr static auto dyn =
            uvl::detail::dynamic_extents_indices_v<decltype(E)>;
        static_assert(dyn.size() == 1);
        static_assert(dyn[0] == 1);
        auto val = uvl::detail::dynamic_extents(E);
        static_assert(val.size() == 1);
        CHECK(val[0] == 19);
    }
    {
        uvl::extents<std::dynamic_extent, 5, std::dynamic_extent> E(3, 5);
        constexpr static auto dyn =
            uvl::detail::dynamic_extents_indices_v<decltype(E)>;
        static_assert(dyn.size() == 2);
        static_assert(dyn[0] == 0);
        static_assert(dyn[1] == 2);
        auto val = uvl::detail::dynamic_extents(E);
        static_assert(val.size() == 2);
        CHECK(val[0] == 3);
        CHECK(val[1] == 5);
    }
}
TEST_CASE("test_extents_swizzle", "[extents][dense]") {
    {
        uvl::extents<5, std::dynamic_extent> E(19);
        auto SE = uvl::detail::swizzle_extents(
            E, std::integer_sequence<uvl::index_type, 0>{});

        REQUIRE(SE.rank() == 1);
        CHECK(SE == uvl::dextents<1>(5));
    }
    {
        uvl::extents<5, std::dynamic_extent> E(19);
        auto SE = uvl::detail::swizzle_extents(
            E, std::integer_sequence<uvl::index_type, 1>{});

        REQUIRE(SE.rank() == 1);
        CHECK(SE.static_extent(0) == std::dynamic_extent);
        CHECK(SE == uvl::dextents<1>(19));
    }
    {
        uvl::extents<5, std::dynamic_extent> E(19);
        auto SE = uvl::detail::swizzle_extents(
            E, std::integer_sequence<uvl::index_type, 1, 0>{});

        REQUIRE(SE.rank() == 2);
        CHECK(SE.static_extent(0) == std::dynamic_extent);
        CHECK(SE.static_extent(1) == 5);
        CHECK(SE == uvl::dextents<2>(19, 5));
    }

    {
        uvl::extents<std::dynamic_extent, 5, std::dynamic_extent> E(3, 6);
        REQUIRE(E == uvl::dextents<3>(3, 5, 6));

        auto e3 = uvl::detail::swizzle_extents(
            E, std::integer_sequence<uvl::index_type, 0>{});
        static_assert(decltype(e3)::rank() == 1);
        static_assert(decltype(e3)::static_extent(0) == std::dynamic_extent);
        CHECK(e3.extent(0) == 3);

        CHECK(uvl::detail::swizzle_extents(
                  E, std::integer_sequence<uvl::index_type, 0>{}) ==
              uvl::create_dextents(3));
        CHECK(uvl::detail::swizzle_extents(
                  E, std::integer_sequence<uvl::index_type, 1>{}) ==
              uvl::create_dextents(5));
        CHECK(uvl::detail::swizzle_extents(
                  E, std::integer_sequence<uvl::index_type, 0, 1>{}) ==
              uvl::create_dextents(3, 5));
        CHECK(uvl::detail::swizzle_extents(
                  E, std::integer_sequence<uvl::index_type, 1, 0>{}) ==
              uvl::create_dextents(5, 3));

        CHECK(uvl::detail::swizzle_extents(
                  E, std::integer_sequence<uvl::index_type, 1, 1>{}) ==
              uvl::create_dextents(5, 5));

        CHECK(uvl::detail::swizzle_extents(
                  E, std::integer_sequence<uvl::index_type, 2, 1, 0>{}) ==
              uvl::create_dextents(6, 5, 3));

        uvl::detail::ExtentsSwizzler<2, 1, 0, 1> s;
        auto se = s.swizzle_extents(E);
        CHECK(se == uvl::create_dextents(6, 5, 3, 5));

        auto arr = s.swizzle(10, 11, 12);
        REQUIRE(arr.size() == 4);
        auto arr2 = s.swizzle(std::make_tuple(10, 11, 12));
        REQUIRE(arr2.size() == 4);
        CHECK(arr == arr2);
        CHECK(arr[0] == 12);
        CHECK(arr[1] == 11);
        CHECK(arr[2] == 10);
        CHECK(arr[3] == 11);
    }
}
