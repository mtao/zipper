
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/unary/AntiSlice.hpp>

#include "../../catch_include.hpp"

using namespace zipper;

TEST_CASE("AntiSlice vector to column matrix", "[expression][antislice]") {
    // Insert dim at position 1: Vector<3> → extents(3, 1)
    using namespace zipper::expression::unary;

    Vector<double, 3> v{1.0, 2.0, 3.0};
    auto &expr = v.expression();
    AntiSlice<decltype(expr), 1> as(expr);

    CHECK(as.extent(0) == 3);
    CHECK(as.extent(1) == 1);

    // Reading: as(i, 0) == v(i)
    CHECK(as.coeff(index_type{0}, index_type{0}) == 1.0);
    CHECK(as.coeff(index_type{1}, index_type{0}) == 2.0);
    CHECK(as.coeff(index_type{2}, index_type{0}) == 3.0);
}

TEST_CASE("AntiSlice vector to row matrix", "[expression][antislice]") {
    // Insert dim at position 0: Vector<3> → extents(1, 3)
    using namespace zipper::expression::unary;

    Vector<double, 3> v{4.0, 5.0, 6.0};
    auto &expr = v.expression();
    AntiSlice<decltype(expr), 0> as(expr);

    CHECK(as.extent(0) == 1);
    CHECK(as.extent(1) == 3);

    CHECK(as.coeff(index_type{0}, index_type{0}) == 4.0);
    CHECK(as.coeff(index_type{0}, index_type{1}) == 5.0);
    CHECK(as.coeff(index_type{0}, index_type{2}) == 6.0);
}

TEST_CASE("AntiSlice scalar to vector", "[expression][antislice]") {
    // Insert dim at position 0 on a rank-0 → rank-1 with extent 1
    // Actually let's use a rank-1 expression and add a dim to get rank-2
    // Use a 1-element vector and add a dimension
    using namespace zipper::expression::unary;

    Vector<double, 1> v{42.0};
    auto &expr = v.expression();
    AntiSlice<decltype(expr), 0> as(expr);

    CHECK(as.extent(0) == 1);
    CHECK(as.extent(1) == 1);
    CHECK(as.coeff(index_type{0}, index_type{0}) == 42.0);
}

TEST_CASE("AntiSlice multiple dims", "[expression][antislice]") {
    // Vector<2> with two dims inserted → rank 3: extents depend on positions
    // Insert at 0 and 2: output rank = 3
    // child_index_map: [dyn, 0, dyn] → child maps to position 1
    // But actually InsertedDims=0,2 means positions 0 and 2 are inserted
    // Output: (1, child_extent(0), 1) = (1, 2, 1)
    using namespace zipper::expression::unary;

    Vector<double, 2> v{10.0, 20.0};
    auto &expr = v.expression();
    AntiSlice<decltype(expr), 0, 2> as(expr);

    CHECK(as.extent(0) == 1);
    CHECK(as.extent(1) == 2);
    CHECK(as.extent(2) == 1);

    CHECK(as.coeff(index_type{0}, index_type{0}, index_type{0}) == 10.0);
    CHECK(as.coeff(index_type{0}, index_type{1}, index_type{0}) == 20.0);
}

TEST_CASE("AntiSlice writable", "[expression][antislice]") {
    // Non-const expression → should be writable (assignable)
    using namespace zipper::expression::unary;

    Vector<double, 3> v{1.0, 2.0, 3.0};
    auto &expr = v.expression();
    AntiSlice<decltype(expr), 1> as(expr);

    // Write through AntiSlice
    as.coeff_ref(index_type{0}, index_type{0}) = 100.0;
    as.coeff_ref(index_type{2}, index_type{0}) = 300.0;

    // Verify the underlying vector was modified
    CHECK(v(0) == 100.0);
    CHECK(v(1) == 2.0);
    CHECK(v(2) == 300.0);
}

TEST_CASE("AntiSlice const not writable", "[expression][antislice]") {
    // const expression → coeff works, coeff_ref should not compile
    // We just check coeff access works on a const-qualified child
    using namespace zipper::expression::unary;

    const Vector<double, 2> v{7.0, 8.0};
    auto &expr = v.expression();
    AntiSlice<const decltype(expr), 1> as(expr);

    CHECK(as.coeff(index_type{0}, index_type{0}) == 7.0);
    CHECK(as.coeff(index_type{1}, index_type{0}) == 8.0);
}

TEST_CASE("AntiSlice matrix add dim", "[expression][antislice]") {
    // Matrix<2,2> → rank-3 by inserting dim at position 2
    // Output extents: (2, 2, 1)
    using namespace zipper::expression::unary;

    Matrix<double, 2, 2> M{{1.0, 2.0}, {3.0, 4.0}};
    auto &expr = M.expression();
    AntiSlice<decltype(expr), 2> as(expr);

    CHECK(as.extent(0) == 2);
    CHECK(as.extent(1) == 2);
    CHECK(as.extent(2) == 1);

    CHECK(as.coeff(index_type{0}, index_type{0}, index_type{0}) == 1.0);
    CHECK(as.coeff(index_type{0}, index_type{1}, index_type{0}) == 2.0);
    CHECK(as.coeff(index_type{1}, index_type{0}, index_type{0}) == 3.0);
    CHECK(as.coeff(index_type{1}, index_type{1}, index_type{0}) == 4.0);
}
