
#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <zipper/detail/extents/all_extents_indices.hpp>
#include <zipper/Vector.hpp>

TEST_CASE("test_all_extents", "[storage][dense]") {
    for (const auto& ind : zipper::detail::extents::all_extents_indices(3, 4)) {
        // auto v = ind | ranges::views::transform([](auto i) { return 2 * i;
        // });

        spdlog::info("{} {}", std::get<0>(ind), std::get<1>(ind));
    }
}
TEST_CASE("test_vector_iterate", "[vector][dense]") {

    zipper::Vector<double,5> a = {0,1,2,3,4};
    zipper::Vector<double,std::dynamic_extent> b = {2,3,4,5};

    CHECK(a(0) == 0);
    CHECK(a(1) == 1);
    CHECK(a(2) == 2);
    CHECK(a(3) == 3);
    CHECK(a(4) == 4);

    a = {5,4,3,2,1};
    CHECK(a(0) == 5);
    CHECK(a(1) == 4);
    CHECK(a(2) == 3);
    CHECK(a(3) == 2);
    CHECK(a(4) == 1);

    REQUIRE(b.extent(0) == 4);
    CHECK(b(0) == 2);
    CHECK(b(1) == 3);
    CHECK(b(2) == 4);
    CHECK(b(3) == 5);

    b = {4,9,5};
    REQUIRE(b.extent(0) == 3);
    CHECK(b(0) == 4);
    CHECK(b(1) == 9);
    CHECK(b(2) == 5);
}
