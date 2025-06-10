#include <catch2/catch_all.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/determinant.hpp>

TEST_CASE("test_matrix_determinant", "[utils][determinant]") {
    REQUIRE(zipper::detail::extents::detail::max_dim<zipper::extents<3, 4>>() ==
            4);
    REQUIRE(zipper::detail::extents::detail::max_dim<zipper::extents<2, 4>>() ==
            4);
    REQUIRE(zipper::detail::extents::detail::max_dim<zipper::extents<4, 2>>() ==
            4);
    // static assert preents these
    // REQUIRE(zipper::detail::extents::detail::max_dim<
    //            zipper::extents<std::dynamic_extent, 2>>() == 2);
    // REQUIRE(zipper::detail::extents::detail::max_dim<
    //            zipper::extents<4, std::dynamic_extent>>() == 4);

    REQUIRE(zipper::detail::extents::detail::max_dim(zipper::extents<3, 4>{}) ==
            4);
    REQUIRE(zipper::detail::extents::detail::max_dim(zipper::extents<2, 4>{}) ==
            4);
    REQUIRE(zipper::detail::extents::detail::max_dim(zipper::extents<4, 2>{}) ==
            4);
    REQUIRE(zipper::detail::extents::detail::max_dim(
                zipper::extents<std::dynamic_extent, 2>{1}) == 2);
    REQUIRE(zipper::detail::extents::detail::max_dim(
                zipper::extents<4, std::dynamic_extent>{1}) == 4);

    REQUIRE(zipper::detail::extents::detail::max_dim(
                zipper::extents<std::dynamic_extent, 2>{10}) == 10);
    REQUIRE(zipper::detail::extents::detail::max_dim(
                zipper::extents<4, std::dynamic_extent>{10}) == 10);

    REQUIRE(zipper::detail::extents::detail::max_dim(
                zipper::extents<std::dynamic_extent, std::dynamic_extent>{
                    3, 5}) == 5);
    REQUIRE(zipper::detail::extents::detail::max_dim(
                zipper::extents<std::dynamic_extent, std::dynamic_extent>{
                    5, 3}) == 5);

    CHECK(zipper::detail::extents::is_cubic(zipper::extents<4, 4>{}));
    CHECK_FALSE(zipper::detail::extents::is_cubic(zipper::extents<3, 4>{}));
    CHECK_FALSE(zipper::detail::extents::is_cubic(zipper::extents<5, 4>{}));

    CHECK(zipper::detail::extents::is_cubic(
        zipper::extents<std::dynamic_extent, 4>{4}));
    CHECK_FALSE(zipper::detail::extents::is_cubic(
        zipper::extents<std::dynamic_extent, 4>{5}));
    CHECK(zipper::detail::extents::is_cubic(
        zipper::extents<std::dynamic_extent, std::dynamic_extent>{5, 5}));
    CHECK_FALSE(zipper::detail::extents::is_cubic(
        zipper::extents<std::dynamic_extent, std::dynamic_extent>{5, 4}));
    CHECK_FALSE(zipper::detail::extents::is_cubic(
        zipper::extents<std::dynamic_extent, std::dynamic_extent>{4, 5}));

    zipper::Matrix<double, 3, 3> x;
    x.col(0) = {1, 0, 0};
    x.col(1) = {0, 1, 0};
    x.col(2) = {0, 0, 1};

    CHECK(zipper::utils::determinant(x) == 1);
    x.col(1) = {0, 0, 1};
    x.col(2) = {0, 1, 0};

    CHECK(zipper::utils::determinant(x) == -1);

    x.col(0) = {0, 1, 0};
    x.col(1) = {0, 1, 0};
    x.col(2) = {1, 0, 0};
    CHECK(zipper::utils::determinant(x) == 0);
}
