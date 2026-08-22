

#include "../../catch_include.hpp"
#include <zipper/utils/extents/all_extents_indices.hpp>
#include <zipper/expression/detail/intersect_nonzeros.hpp>
#include <zipper/Vector.hpp>
using namespace zipper;

TEST_CASE("all_extents_indices_visits_each_index", "[extents][iteration]") {
    std::vector<std::tuple<index_type, index_type>> indices;
    for (const auto& ind : zipper::utils::extents::all_extents_indices(3, 4)) {
        indices.push_back(ind);
    }

    REQUIRE(indices.size() == 12);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 4; ++j) {
            CHECK(indices[4 * i + j] == std::tuple{i, j});
        }
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

TEST_CASE("test_iterate_nonzeros", "[vector][nonzeros]") {


    std::vector<index_type> a = {0,3,4,6};
    std::vector<index_type> b = {3,4,5};

    {
    zipper::expression::detail::intersect_nonzeros innz(a,b);

    STATIC_CHECK(std::ranges::range<std::decay_t<decltype(a)>>);
    STATIC_CHECK(std::ranges::range<std::decay_t<decltype(b)>>);
    STATIC_CHECK(std::ranges::range<std::decay_t<decltype(innz)>>);

    auto res = std::ranges::views::all(innz) | std::ranges::to<std::vector>();
    REQUIRE(res.size() == 2);
    CHECK(res[0] == 3);
    CHECK(res[1] == 4);
    }
    {
    zipper::expression::detail::intersect_nonzeros innz(a,a);
    auto res = std::ranges::views::all(innz) | std::ranges::to<std::vector>();
    REQUIRE(res.size() == 4);
    CHECK(res[0] == 0);
    CHECK(res[1] == 3);
    CHECK(res[2] == 4);
    CHECK(res[3] == 6);
    }
}
