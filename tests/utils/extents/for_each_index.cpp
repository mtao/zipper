#include "../../catch_include.hpp"
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/types.hpp>
#include <zipper/utils/extents/for_each_index.hpp>
#include <array>
#include <utility>
#include <vector>

namespace {
using zipper::index_type;
using Pair = std::pair<index_type, index_type>;

template <typename Extents>
auto visit_row_major(const Extents &e) {
    std::vector<Pair> out;
    zipper::utils::extents::for_each_index<zipper::storage::layout_right>(
        e, [&](index_type i, index_type j) { out.emplace_back(i, j); });
    return out;
}
template <typename Extents>
auto visit_col_major(const Extents &e) {
    std::vector<Pair> out;
    zipper::utils::extents::for_each_index<zipper::storage::layout_left>(
        e, [&](index_type i, index_type j) { out.emplace_back(i, j); });
    return out;
}
auto expected_row_major(index_type rows, index_type cols) {
    std::vector<Pair> out;
    for (index_type i = 0; i < rows; ++i)
        for (index_type j = 0; j < cols; ++j) out.emplace_back(i, j);
    return out;
}
auto expected_col_major(index_type rows, index_type cols) {
    std::vector<Pair> out;
    for (index_type j = 0; j < cols; ++j)
        for (index_type i = 0; i < rows; ++i) out.emplace_back(i, j);
    return out;
}
} // namespace

// The rank-1/2 loops carry an unroll hint; static, dynamic and mixed extents
// (including sizes above the hint's factor) must all visit the same indices in
// the same order.
TEST_CASE("for_each_index_order_static_and_dynamic_extents",
          "[extents][iteration]") {
    using zipper::dynamic_extent;
    constexpr zipper::index_type big = 11; // larger than the unroll factor (8)

    SECTION("static 4x4") {
        zipper::extents<4, 4> e;
        CHECK(visit_row_major(e) == expected_row_major(4, 4));
        CHECK(visit_col_major(e) == expected_col_major(4, 4));
    }
    SECTION("static 3x5 (non-square)") {
        zipper::extents<3, 5> e;
        CHECK(visit_row_major(e) == expected_row_major(3, 5));
        CHECK(visit_col_major(e) == expected_col_major(3, 5));
    }
    SECTION("mixed static/dynamic") {
        zipper::extents<4, dynamic_extent> e(7);
        CHECK(visit_row_major(e) == expected_row_major(4, 7));
        CHECK(visit_col_major(e) == expected_col_major(4, 7));
        zipper::extents<dynamic_extent, 2> f(6);
        CHECK(visit_row_major(f) == expected_row_major(6, 2));
        CHECK(visit_col_major(f) == expected_col_major(6, 2));
    }
    SECTION("dynamic") {
        zipper::dextents<2> e(3, 4);
        CHECK(visit_row_major(e) == expected_row_major(3, 4));
        CHECK(visit_col_major(e) == expected_col_major(3, 4));
    }
    SECTION("static larger than the unroll factor") {
        zipper::extents<big, 2> e;
        CHECK(visit_row_major(e) == expected_row_major(big, 2));
        CHECK(visit_col_major(e) == expected_col_major(big, 2));
    }
    SECTION("zero extents") {
        CHECK(visit_row_major(zipper::extents<0, 4>{}).empty());
        CHECK(visit_col_major(zipper::extents<4, 0>{}).empty());
        CHECK(visit_row_major(zipper::dextents<2>(0, 3)).empty());
    }
    SECTION("rank 1") {
        std::vector<index_type> s, d;
        zipper::utils::extents::for_each_index<zipper::storage::layout_right>(
            zipper::extents<5>{}, [&](index_type i) { s.push_back(i); });
        zipper::utils::extents::for_each_index<zipper::storage::layout_left>(
            zipper::dextents<1>(5), [&](index_type i) { d.push_back(i); });
        CHECK(s == std::vector<index_type>{0, 1, 2, 3, 4});
        CHECK(d == s);
    }
}

TEST_CASE("for_each_index_order_rank3", "[extents][iteration]") {
    using Triple = std::array<index_type, 3>;
    auto check = [](const auto &e, index_type a, index_type b, index_type c) {
        std::vector<Triple> row, col, row_expected, col_expected;
        zipper::utils::extents::for_each_index<zipper::storage::layout_right>(
            e, [&](index_type i, index_type j, index_type k) {
                row.push_back({i, j, k});
            });
        zipper::utils::extents::for_each_index<zipper::storage::layout_left>(
            e, [&](index_type i, index_type j, index_type k) {
                col.push_back({i, j, k});
            });
        for (index_type i = 0; i < a; ++i)
            for (index_type j = 0; j < b; ++j)
                for (index_type k = 0; k < c; ++k) row_expected.push_back({i, j, k});
        for (index_type k = 0; k < c; ++k)
            for (index_type j = 0; j < b; ++j)
                for (index_type i = 0; i < a; ++i) col_expected.push_back({i, j, k});
        CHECK(row == row_expected);
        CHECK(col == col_expected);
    };
    check(zipper::extents<2, 3, 4>{}, 2, 3, 4);
    check(zipper::dextents<3>(3, 2, 5), 3, 2, 5);
    check(zipper::extents<2, zipper::dynamic_extent, 3>(4), 2, 4, 3);
}

TEST_CASE("for_each_index_rank0_calls_once", "[extents][iteration]") {
    int calls = 0;
    zipper::utils::extents::for_each_index<zipper::storage::layout_right>(zipper::extents<>{}, [&] { ++calls; });
    zipper::utils::extents::for_each_index<zipper::storage::layout_left>(zipper::extents<>{}, [&] { ++calls; });
    CHECK(calls == 2);
}

TEST_CASE("for_each_index_expanded_assignment_values", "[extents][iteration]") {
    // Assignments of small static objects go through the unrolled loops.
    zipper::Matrix<double, 4, 4, false> a;
    for (index_type j = 0; j < 4; ++j)
        for (index_type i = 0; i < 4; ++i) a(i, j) = double(10 * i + j);
    zipper::Matrix<double, 4, 4, true> b = a;
    zipper::Vector<double, 4> row = a.row(2);
    for (index_type j = 0; j < 4; ++j) {
        CHECK(row(j) == double(20 + j));
        for (index_type i = 0; i < 4; ++i) CHECK(b(i, j) == a(i, j));
    }
}

TEST_CASE("for_each_index_layout_preferences", "[extents][iteration]") {
    zipper::extents<2, 3> e;
    auto order = [&]<typename L>() {
        std::vector<Pair> out;
        zipper::utils::extents::for_each_index<L>(
            e, [&](index_type i, index_type j) { out.emplace_back(i, j); });
        return out;
    };
    using zipper::detail::DenseLayoutPreference;
    using zipper::detail::NoLayoutPreference;
    using zipper::storage::layout_left;
    using zipper::storage::layout_right;
    CHECK(order.template operator()<NoLayoutPreference>() == expected_row_major(2, 3));
    CHECK(order.template operator()<DenseLayoutPreference<layout_right>>() ==
          expected_row_major(2, 3));
    CHECK(order.template operator()<DenseLayoutPreference<layout_left>>() ==
          expected_col_major(2, 3));
}
