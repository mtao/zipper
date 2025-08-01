
#include <zipper/utils/format.hpp>

#include <zipper/Matrix.hpp>
//#include <zipper/Vector.hpp>
#include "../catch_include.hpp"

TEST_CASE("test_format_vector", "[storage][dense]") {
    zipper::Vector<double, 3> x({1, 2, 0});
    fmt::print("{}\n",x);
    fmt::print("{}\n",x.head<2>());
}
TEST_CASE("test_format_matrix", "[storage][dense]") {
    zipper::Matrix<double, 3, 3> x;
    x.col(0) = {0, 1, 2};
    x.col(1) = {-3, 3, 0};
    x.col(2) = {100, 4, 1};

    fmt::print("{}\n",x);

}
