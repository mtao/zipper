


#include "fmt_include.hpp"
#include "catch_include.hpp"
#include <iostream>
#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/Form.hpp>
// #include <zipper/Vector.hpp>



struct A {
    zipper::Array<double,3> u = {};
    zipper::Vector<double,3> x = {};
    zipper::Matrix<double,3,3> y = {};
    zipper::Tensor<double,3,3> z = {};
    zipper::Form<double,3,3> w = {};
};
TEST_CASE("test_vector_struct", "[matrix][storage][dense]") {

    A a;
    a.x = {3,4,1};
    A b = a;
    CHECK(b.x == a.x);

    A c(a);
    CHECK(c.x == a.x);

    b.x(1) = 3;
    c = std::move(b);
    CHECK(c.x(0) == a.x(0));
    CHECK(c.x(1) == b.x(1));
    CHECK(c.x(2) == a.x(2));

}
