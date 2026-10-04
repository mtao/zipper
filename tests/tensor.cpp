// Tensor assignment: every value category, across layouts and extents
// types, and from views.

#include <zipper/Tensor.hpp>

#include "catch_include.hpp"

using namespace zipper;

namespace {
template <typename T>
void fill(T &t, double offset) {
    for (index_type i = 0; i < t.extent(0); ++i)
        for (index_type j = 0; j < t.extent(1); ++j)
            for (index_type k = 0; k < t.extent(2); ++k)
                t(i, j, k) = offset + 100.0 * i + 10.0 * j + k;
}

template <typename A, typename B>
void check_equal(const A &a, const B &b) {
    REQUIRE(a.extent(0) == b.extent(0));
    REQUIRE(a.extent(1) == b.extent(1));
    REQUIRE(a.extent(2) == b.extent(2));
    for (index_type i = 0; i < a.extent(0); ++i)
        for (index_type j = 0; j < a.extent(1); ++j)
            for (index_type k = 0; k < a.extent(2); ++k)
                CHECK(a(i, j, k) == b(i, j, k));
}

using DTensor = Tensor<double, dynamic_extent, dynamic_extent, dynamic_extent>;
using DTensorL = Tensor_<double, dextents<3>, false>;
} // namespace

TEST_CASE("tensor_assign_same_type", "[tensor][assign]") {
    STATIC_CHECK(std::is_copy_assignable_v<DTensor>);
    STATIC_CHECK(std::is_move_assignable_v<DTensor>);
    STATIC_CHECK(std::is_copy_assignable_v<Tensor<double, 2, 3, 4>>);
    STATIC_CHECK(std::is_trivially_copyable_v<Tensor<double, 2, 3, 4>>);

    DTensor a(2, 3, 4);
    fill(a, 0.0);

    SECTION("lvalue") {
        DTensor b(2, 3, 4);
        b = a;
        check_equal(a, b);
        b(0, 0, 0) = -1.0; // b is a copy, not a view
        CHECK(a(0, 0, 0) == 0.0);
    }
    SECTION("const lvalue") {
        const DTensor &ca = a;
        DTensor b(2, 3, 4);
        b = ca;
        check_equal(a, b);
    }
    SECTION("rvalue") {
        DTensor tmp = a;
        DTensor b(2, 3, 4);
        b = std::move(tmp);
        check_equal(a, b);
    }
    SECTION("resizes a dynamic destination") {
        DTensor b(1, 1, 1);
        b = a;
        check_equal(a, b);
    }
    SECTION("static extents") {
        Tensor<double, 2, 3, 4> s, t;
        fill(s, 1.0);
        t = s;
        check_equal(s, t);
    }
}

TEST_CASE("tensor_assign_across_types", "[tensor][assign]") {
    DTensor a(2, 3, 4);
    fill(a, 0.0);

    SECTION("other layout") {
        DTensorL b(2, 3, 4);
        b = a;
        check_equal(a, b);
        DTensor c(2, 3, 4);
        c = b;
        check_equal(a, c);
    }
    SECTION("static <-> dynamic extents") {
        Tensor<double, 2, 3, 4> s;
        s = a;
        check_equal(a, s);
        DTensor d(2, 3, 4);
        d = s;
        check_equal(a, d);
    }
    SECTION("from views") {
        // P(k, i, j) = a(i, j, k)
        DTensor p(4, 2, 3);
        p = a.swizzle<TensorBase, 2, 0, 1>();
        for (index_type i = 0; i < 2; ++i)
            for (index_type j = 0; j < 3; ++j)
                for (index_type k = 0; k < 4; ++k) CHECK(p(k, i, j) == a(i, j, k));

        DTensor s(1, 3, 2);
        s = a.slice(zipper::slice(1, 1), full_extent_t{}, zipper::slice(2, 2));
        for (index_type j = 0; j < 3; ++j)
            for (index_type k = 0; k < 2; ++k) CHECK(s(0, j, k) == a(1, j, 2 + k));
    }
    SECTION("into a view") {
        DTensor b(2, 3, 4);
        fill(b, 1000.0);
        const DTensor b0 = b;
        // (A view assigned from a view of the exact same type is ambiguous,
        // for matrices too — a general view issue, not tensor-specific; a
        // const source gives a distinct view type.)
        const DTensor &ca = a;
        b.slice(full_extent_t{}, zipper::slice(0, 2), full_extent_t{}) =
            ca.slice(full_extent_t{}, zipper::slice(1, 2), full_extent_t{});
        for (index_type i = 0; i < 2; ++i)
            for (index_type j = 0; j < 3; ++j)
                for (index_type k = 0; k < 4; ++k)
                    CHECK(b(i, j, k) == (j < 2 ? a(i, j + 1, k) : b0(i, j, k)));
    }
    SECTION("compound") {
        DTensor b = a;
        b += a;
        for (index_type i = 0; i < 2; ++i)
            for (index_type j = 0; j < 3; ++j)
                for (index_type k = 0; k < 4; ++k) CHECK(b(i, j, k) == 2.0 * a(i, j, k));
    }
}
