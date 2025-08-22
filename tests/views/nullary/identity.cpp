
#include <iostream>
#include <zipper/views/nullary/IdentityView.hpp>

#include "../../catch_include.hpp"
#include "../../fmt_include.hpp"

using namespace zipper::views::nullary;
using namespace zipper;

TEST_CASE("test_identity", "[views][nullary]") {

    IdentityView<double,std::dynamic_extent> a( create_dextents(3));
    IdentityView<double,3,3> b;
    IdentityView<double> c;


    REQUIRE(a.extents().rank() == 1);
    CHECK(a.extent(0)== 3);
    REQUIRE(b.extents().rank() == 2);
    CHECK(b.extent(0)== 3);
    CHECK(b.extent(1)== 3);

    REQUIRE(c.extents().rank() == 0);

    CHECK(a(0) == 1);
    CHECK(a(1) == 0);
    CHECK(a(2) == 0);

    CHECK(b(0,0) == 1);
    CHECK(b(0,1) == 0);
    CHECK(b(0,2) == 0);
    CHECK(b(1,0) == 0);
    CHECK(b(1,1) == 1);
    CHECK(b(1,2) == 0);

    // for an "infinite" view we can pass in random values
    CHECK(c(0) == 1);
    CHECK(c(2) == 0);
    CHECK(c(1,2) == 0);
    CHECK(c(2,2) == 1);
    CHECK(c(2,3,1,2) == 0);
    CHECK(c(3,3,3,3) == 1);
    CHECK(c() == 1);

}
