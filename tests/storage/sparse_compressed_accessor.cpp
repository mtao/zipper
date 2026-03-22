
#include "../catch_include.hpp"
#include <zipper/storage/SparseCompressedAccessor.hpp>

using namespace zipper;

TEST_CASE("sparse_compressed_accessor_from_coo", "[sparse][compressed]") {
    storage::SparseCoordinateAccessor<double, extents<4, 5>> coo;
    coo.emplace(0, 1) = 1.0;
    coo.emplace(0, 3) = 2.0;
    coo.emplace(1, 2) = 3.0;
    coo.emplace(2, 0) = 4.0;
    coo.emplace(2, 3) = 5.0;
    coo.compress();

    storage::SparseCompressedAccessor<double, extents<4, 5>> A(coo);

    SECTION("coeff returns stored values") {
        CHECK(A.coeff(0, 1) == 1.0);
        CHECK(A.coeff(0, 3) == 2.0);
        CHECK(A.coeff(1, 2) == 3.0);
        CHECK(A.coeff(2, 0) == 4.0);
        CHECK(A.coeff(2, 3) == 5.0);
    }

    SECTION("coeff returns 0 for missing entries") {
        CHECK(A.coeff(0, 0) == 0.0);
        CHECK(A.coeff(0, 2) == 0.0);
        CHECK(A.coeff(1, 0) == 0.0);
        CHECK(A.coeff(3, 4) == 0.0);
    }
}

TEST_CASE("sparse_compressed_accessor_coeff_ref", "[sparse][compressed]") {
    storage::SparseCoordinateAccessor<double, extents<3, 3>> coo;
    coo.emplace(0, 0) = 1.0;
    coo.emplace(1, 1) = 2.0;
    coo.emplace(2, 2) = 3.0;
    coo.compress();

    storage::SparseCompressedAccessor<double, extents<3, 3>> A(coo);

    SECTION("coeff_ref on existing entry") {
        A.coeff_ref(1, 1) = 99.0;
        CHECK(A.coeff(1, 1) == 99.0);
    }

    SECTION("coeff_ref on missing entry throws") {
        CHECK_THROWS_AS(A.coeff_ref(0, 1), std::invalid_argument);
    }

    SECTION("const_coeff_ref on existing entry") {
        const auto &val = A.const_coeff_ref(2, 2);
        CHECK(val == 3.0);
    }

    SECTION("const_coeff_ref on missing entry throws") {
        CHECK_THROWS_AS(A.const_coeff_ref(0, 1), std::invalid_argument);
    }
}

TEST_CASE("sparse_compressed_accessor_const_type", "[sparse][compressed]") {
    storage::SparseCoordinateAccessor<double, extents<3, 3>> coo;
    coo.emplace(0, 0) = 1.0;
    coo.emplace(1, 1) = 2.0;
    coo.compress();

    // const ValueType → read-only
    storage::SparseCompressedAccessor<const double, extents<3, 3>> A(coo);

    CHECK(A.coeff(0, 0) == 1.0);
    CHECK(A.coeff(1, 0) == 0.0);

    // coeff_ref should not compile for const ValueType — verified by trait
    using traits = expression::detail::ExpressionTraits<decltype(A)>;
    STATIC_REQUIRE(traits::access_features.is_const);
    STATIC_REQUIRE(!traits::access_features.is_reference);
}

TEST_CASE("sparse_compressed_accessor_index_set_rank2",
          "[sparse][compressed][index_set]") {
    storage::SparseCoordinateAccessor<double, extents<4, 5>> coo;
    coo.emplace(0, 1) = 1.0;
    coo.emplace(0, 3) = 2.0;
    coo.emplace(1, 2) = 3.0;
    coo.emplace(2, 0) = 4.0;
    coo.emplace(2, 3) = 5.0;
    coo.compress();

    storage::SparseCompressedAccessor<double, extents<4, 5>> A(coo);

    SECTION("col indices for row 0 = {1, 3}") {
        auto cols = A.index_set<1>(0);
        REQUIRE(cols.size() == 2);
        auto it = cols.begin();
        CHECK(*it++ == 1);
        CHECK(*it++ == 3);
    }

    SECTION("col indices for row 1 = {2}") {
        auto cols = A.index_set<1>(1);
        REQUIRE(cols.size() == 1);
        CHECK(*cols.begin() == 2);
    }

    SECTION("col indices for row 2 = {0, 3}") {
        auto cols = A.index_set<1>(2);
        REQUIRE(cols.size() == 2);
        auto it = cols.begin();
        CHECK(*it++ == 0);
        CHECK(*it++ == 3);
    }

    SECTION("col indices for empty row 3 = {}") {
        auto cols = A.index_set<1>(3);
        CHECK(cols.empty());
    }

    SECTION("row indices for col 0 = {2}") {
        auto rows = A.index_set<0>(0);
        REQUIRE(rows.size() == 1);
        CHECK(*rows.begin() == 2);
    }

    SECTION("row indices for col 3 = {0, 2}") {
        auto rows = A.index_set<0>(3);
        REQUIRE(rows.size() == 2);
        auto it = rows.begin();
        CHECK(*it++ == 0);
        CHECK(*it++ == 2);
    }

    SECTION("row indices for empty col 4 = {}") {
        auto rows = A.index_set<0>(4);
        CHECK(rows.empty());
    }

    SECTION("convenience aliases") {
        CHECK(A.col_range_for_row(0).size() == 2);
        CHECK(A.row_range_for_col(3).size() == 2);
    }
}

TEST_CASE("sparse_compressed_accessor_index_set_rank1",
          "[sparse][compressed][index_set]") {
    storage::SparseCoordinateAccessor<double, extents<10>> coo;
    coo.emplace(2) = 1.0;
    coo.emplace(5) = 2.0;
    coo.emplace(7) = 3.0;
    coo.compress();

    storage::SparseCompressedAccessor<double, extents<10>> A(coo);

    auto idx = A.index_set<0>();
    REQUIRE(idx.size() == 3);
    auto it = idx.begin();
    CHECK(*it++ == 2);
    CHECK(*it++ == 5);
    CHECK(*it++ == 7);
    CHECK(idx.contains(5));
    CHECK(!idx.contains(0));

    SECTION("convenience alias") {
        CHECK(A.nonzero_segment().size() == 3);
    }
}

TEST_CASE("sparse_compressed_accessor_traits", "[sparse][compressed]") {
    using SCA =
        storage::SparseCompressedAccessor<double, extents<3, 3>>;
    using traits = expression::detail::ExpressionTraits<SCA>;
    STATIC_REQUIRE(traits::has_index_set);
    STATIC_REQUIRE(traits::has_known_zeros);
    STATIC_REQUIRE(!traits::access_features.is_reference);
    STATIC_REQUIRE(!traits::access_features.is_const);
}

TEST_CASE("sparse_compressed_accessor_from_data",
          "[sparse][compressed]") {
    // Build compressed data directly
    storage::detail::SparseCompressedData<double, 1> data;
    data.insert_back(0, 0) = 1.0;
    data.insert_back(1, 1) = 2.0;

    storage::SparseCompressedAccessor<double, extents<3, 3>> A(
        std::move(data));
    CHECK(A.coeff(0, 0) == 1.0);
    CHECK(A.coeff(1, 1) == 2.0);
    CHECK(A.coeff(0, 1) == 0.0);
}

TEST_CASE("sparse_compressed_accessor_empty", "[sparse][compressed]") {
    storage::SparseCompressedAccessor<double, extents<3, 3>> A;
    CHECK(A.coeff(0, 0) == 0.0);
    CHECK(A.coeff(2, 2) == 0.0);

    auto cols = A.index_set<1>(0);
    CHECK(cols.empty());

    auto rows = A.index_set<0>(0);
    CHECK(rows.empty());
}
