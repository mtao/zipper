

#include "../catch_include.hpp"
#include <zipper/views/unary/SwizzleView.hpp>
// #include <zipper/Vector.hpp>

TEST_CASE("test_swizzle_pair", "[swizzle]") {
    using Swizzler = zipper::detail::extents::ExtentsSwizzler<1, 0>;

    REQUIRE(Swizzler::size == 2);
    REQUIRE(Swizzler::valid_indices_rank == 2);
    REQUIRE(Swizzler::valid_internal_indices.size() == 2);
    CHECK(Swizzler::valid_internal_indices[0] == 1);
    CHECK(Swizzler::valid_internal_indices[1] == 0);

    auto a = Swizzler::unswizzle(4, 5);
    CHECK(Swizzler::_unswizzle_single_index<0>(std::make_tuple(4, 5)) == 5);
    CHECK(Swizzler::_unswizzle_single_index<1>(std::make_tuple(4, 5)) == 4);
    REQUIRE(a.size() == 2);
    CHECK(a[0] == 5);
    CHECK(a[1] == 4);
}
TEST_CASE("test_swizzle_lift", "[swizzle]") {
    {
        using Swizzler =
            zipper::detail::extents::ExtentsSwizzler<std::dynamic_extent, 0>;

        REQUIRE(Swizzler::size == 2);
        REQUIRE(Swizzler::valid_indices_rank == 1);
        REQUIRE(Swizzler::valid_internal_indices.size() == 1);
        CHECK(Swizzler::valid_internal_indices[0] == 1);

        CHECK(Swizzler::swizzle_extents_type::static_extent(
                  Swizzler::valid_internal_indices[0]) == 0);

        CHECK(Swizzler::_unswizzle_single_index<0>(std::make_tuple(4, 5)) == 5);
        auto a = Swizzler::unswizzle(4, 5);
        REQUIRE(a.size() == 1);
        CHECK(a[0] == 5);
    }
    {
        using Swizzler =
            zipper::detail::extents::ExtentsSwizzler<0, std::dynamic_extent>;

        REQUIRE(Swizzler::size == 2);
        REQUIRE(Swizzler::valid_indices_rank == 1);
        REQUIRE(Swizzler::valid_internal_indices.size() == 1);
        CHECK(Swizzler::valid_internal_indices[0] == 0);
        CHECK(Swizzler::swizzle_extents_type::static_extent(
                  Swizzler::valid_internal_indices[0]) == 0);
        CHECK(Swizzler::_unswizzle_single_index<0>(std::make_tuple(4, 5)) == 4);

        auto a = Swizzler::unswizzle(4, 5);
        REQUIRE(a.size() == 1);
        CHECK(a[0] == 4);
    }

    {
        using Swizzler =
            zipper::detail::extents::ExtentsSwizzler<1, std::dynamic_extent, 0>;

        REQUIRE(Swizzler::size == 3);
        REQUIRE(Swizzler::valid_indices_rank == 2);
        REQUIRE(Swizzler::valid_internal_indices.size() == 2);
        CHECK(Swizzler::valid_internal_indices[0] == 2);
        CHECK(Swizzler::valid_internal_indices[1] == 0);

        auto a = Swizzler::unswizzle(4, 5, 6);
        CHECK(Swizzler::_unswizzle_single_index<0>(std::make_tuple(4, 5, 6)) ==
              6);
        CHECK(Swizzler::_unswizzle_single_index<1>(std::make_tuple(4, 5, 6)) ==
              4);
        REQUIRE(a.size() == 2);
        CHECK(a[0] == 6);
        CHECK(a[1] == 4);
    }
}
