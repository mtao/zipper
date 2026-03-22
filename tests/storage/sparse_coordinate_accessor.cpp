

#include "../catch_include.hpp"
#include <zipper/storage/SparseCoordinateAccessor.hpp>

// #include "zipper/storage/SpanStorage.hpp"

using namespace zipper;
TEST_CASE("sparse_coordinate_accessor", "[sparse]") {
    auto checker = [](auto& A) {
        A.emplace(3, 3) = 2;
        A.emplace(4, 3) = 3;
        REQUIRE(!A.is_compressed());
        A.emplace(1, 3) = 4;
        A.emplace(4, 4) = 7;
        A.emplace(0, 3) = 9;
        REQUIRE(!A.is_compressed());
        A.compress();
        REQUIRE(A.is_compressed());
        CHECK(A.coeff(3, 2) == 0);
        CHECK(A.coeff_ref(3, 3) == 2);
        CHECK(A.coeff_ref(4, 3) == 3);
        CHECK(A.coeff(3, 3) == 2);
        CHECK(A.coeff(4, 3) == 3);
        CHECK(A.coeff(1, 3) == 4);
        CHECK(A.coeff(4, 4) == 7);
        CHECK(A.coeff(0, 3) == 9);
        A.emplace(0, 3) = 9;
        REQUIRE(!A.is_compressed());
        A.emplace(1, 3) = 5;
        A.emplace(0, 3) = 9;
        A.compress();
        CHECK(A.coeff(1, 3) == 9);
        CHECK(A.coeff(0, 3) == 27);
    };
    storage::SparseCoordinateAccessor<double, extents<5, 5>> A;
    checker(A);
    storage::SparseCoordinateAccessor<double, dextents<2>> B(
        create_dextents(5, 5));
    checker(B);
}

TEST_CASE("sparse_empty", "[sparse]") {
    SECTION("static extents") {
        storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
        // Should start compressed (empty)
        REQUIRE(A.is_compressed());
        // compress() on empty must not crash
        A.compress();
        REQUIRE(A.is_compressed());
        // All coefficients should be zero
        for (index_type i = 0; i < 3; ++i) {
            for (index_type j = 0; j < 3; ++j) {
                CHECK(A.coeff(i, j) == 0.0);
            }
        }
    }
    SECTION("dynamic extents") {
        storage::SparseCoordinateAccessor<double, dextents<2>> A(
            create_dextents(4, 4));
        REQUIRE(A.is_compressed());
        A.compress();
        REQUIRE(A.is_compressed());
        CHECK(A.coeff(0, 0) == 0.0);
    }
}

TEST_CASE("sparse_single_element", "[sparse]") {
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    A.emplace(1, 2) = 42.0;
    A.compress();
    REQUIRE(A.is_compressed());
    CHECK(A.coeff(1, 2) == 42.0);
    CHECK(A.coeff(0, 0) == 0.0);
    CHECK(A.coeff(2, 2) == 0.0);
    // coeff_ref on existing element should work
    CHECK(A.coeff_ref(1, 2) == 42.0);
}

TEST_CASE("sparse_coeff_ref_throws_on_missing", "[sparse]") {
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    A.emplace(0, 0) = 1.0;
    A.compress();
    // coeff_ref on a nonexistent element should throw
    CHECK_THROWS_AS(A.coeff_ref(1, 1), std::invalid_argument);
    CHECK_THROWS_AS(A.coeff_ref(2, 2), std::invalid_argument);
    // const_coeff_ref should also throw on missing
    const auto& cA = A;
    CHECK_THROWS_AS(cA.const_coeff_ref(1, 1), std::invalid_argument);
}

TEST_CASE("sparse_find_before_compress", "[sparse]") {
    // Note: iterator comparison requires compressed state, so we can't
    // directly use find()/coeff() on uncompressed data without hitting
    // the assert in operator<=>.
    // This test verifies that after adding elements and compressing,
    // all values are accessible.
    storage::SparseCoordinateAccessor<double, extents<4, 4>> A;
    A.emplace(2, 3) = 5.0;
    A.emplace(0, 1) = 7.0;
    REQUIRE(!A.is_compressed());
    A.compress();
    REQUIRE(A.is_compressed());
    CHECK(A.coeff(2, 3) == 5.0);
    CHECK(A.coeff(0, 1) == 7.0);
    CHECK(A.coeff(3, 3) == 0.0);
}

TEST_CASE("sparse_iterator", "[sparse]") {
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    A.emplace(2, 0) = 1.0;
    A.emplace(0, 1) = 2.0;
    A.emplace(1, 2) = 3.0;
    A.compress();

    // After compress, iteration should be in lexicographic order
    std::vector<double> vals;
    for (const auto& it : A) {
        vals.push_back(it.value());
    }
    REQUIRE(vals.size() == 3);
    CHECK(vals[0] == 2.0); // (0,1)
    CHECK(vals[1] == 3.0); // (1,2)
    CHECK(vals[2] == 1.0); // (2,0)
}

TEST_CASE("sparse_fully_dense", "[sparse]") {
    // Fill every entry of a small matrix
    storage::SparseCoordinateAccessor<double, extents<2, 2>> A;
    double v = 1.0;
    for (index_type i = 0; i < 2; ++i) {
        for (index_type j = 0; j < 2; ++j) {
            A.emplace(i, j) = v;
            v += 1.0;
        }
    }
    A.compress();
    CHECK(A.coeff(0, 0) == 1.0);
    CHECK(A.coeff(0, 1) == 2.0);
    CHECK(A.coeff(1, 0) == 3.0);
    CHECK(A.coeff(1, 1) == 4.0);
}

TEST_CASE("sparse_multiple_compress_cycles", "[sparse]") {
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    A.emplace(0, 0) = 1.0;
    A.compress();
    CHECK(A.coeff(0, 0) == 1.0);

    // Add more and compress again
    A.emplace(1, 1) = 2.0;
    A.compress();
    CHECK(A.coeff(0, 0) == 1.0);
    CHECK(A.coeff(1, 1) == 2.0);

    // Add duplicate and compress
    A.emplace(0, 0) = 3.0;
    A.compress();
    CHECK(A.coeff(0, 0) == 4.0); // 1.0 + 3.0
    CHECK(A.coeff(1, 1) == 2.0);
}

TEST_CASE("sparse_rank1", "[sparse]") {
    storage::SparseCoordinateAccessor<double, extents<5>> A;
    A.emplace(3) = 10.0;
    A.emplace(1) = 20.0;
    A.compress();
    REQUIRE(A.is_compressed());
    CHECK(A.coeff(0) == 0.0);
    CHECK(A.coeff(1) == 20.0);
    CHECK(A.coeff(3) == 10.0);
    CHECK(A.coeff(4) == 0.0);
}

TEST_CASE("sparse_rank3", "[sparse]") {
    storage::SparseCoordinateAccessor<double, extents<2, 2, 2>> A;
    A.emplace(1, 0, 1) = 5.0;
    A.emplace(0, 1, 0) = 3.0;
    A.compress();
    REQUIRE(A.is_compressed());
    CHECK(A.coeff(0, 0, 0) == 0.0);
    CHECK(A.coeff(0, 1, 0) == 3.0);
    CHECK(A.coeff(1, 0, 1) == 5.0);
    CHECK(A.coeff(1, 1, 1) == 0.0);
}

// --- Edge case tests ---

TEST_CASE("sparse_find_uncompressed_linear_scan", "[sparse]") {
    // Exercise the linear-scan path in find() on uncompressed data.
    // coeff() calls find() internally, and the linear-scan path
    // does not use iterator <=> (which asserts compressed).
    storage::SparseCoordinateAccessor<double, extents<4, 4>> A;
    A.emplace(2, 3) = 5.0;
    A.emplace(0, 1) = 7.0;
    REQUIRE(!A.is_compressed());

    // coeff() should work on uncompressed data via the linear scan
    CHECK(A.coeff(2, 3) == 5.0);
    CHECK(A.coeff(0, 1) == 7.0);
    CHECK(A.coeff(3, 3) == 0.0); // not present
    CHECK(A.coeff(0, 0) == 0.0); // not present
}

TEST_CASE("sparse_duplicate_cancellation", "[sparse]") {
    // Duplicates that sum to zero should still yield zero
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    A.emplace(1, 1) = 5.0;
    A.emplace(1, 1) = -5.0;
    A.compress();
    CHECK(A.coeff(1, 1) == 0.0);
}

TEST_CASE("sparse_negative_values", "[sparse]") {
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    A.emplace(0, 0) = -3.0;
    A.emplace(1, 2) = -7.5;
    A.compress();
    CHECK(A.coeff(0, 0) == -3.0);
    CHECK(A.coeff(1, 2) == -7.5);
    CHECK(A.coeff(2, 2) == 0.0);
}

TEST_CASE("sparse_all_duplicates_same_index", "[sparse]") {
    // Every emplaced entry at the same multi-index
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    for (int i = 0; i < 10; ++i) {
        A.emplace(1, 1) = 1.0;
    }
    A.compress();
    REQUIRE(A.is_compressed());
    CHECK(A.coeff(1, 1) == 10.0); // accumulated sum

    // Iteration should yield exactly one entry
    int count = 0;
    for ([[maybe_unused]] const auto& it : A) {
        ++count;
    }
    CHECK(count == 1);
}

TEST_CASE("sparse_emplace_default_zero", "[sparse]") {
    // emplace() returns a reference initialized to 0; if we don't assign,
    // the value should stay 0
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    [[maybe_unused]] auto& ref = A.emplace(1, 1);
    // Don't assign through ref
    A.compress();
    CHECK(A.coeff(1, 1) == 0.0);
}

TEST_CASE("sparse_integer_value_type", "[sparse]") {
    storage::SparseCoordinateAccessor<int, extents<4, 4>> A;
    A.emplace(0, 0) = 42;
    A.emplace(3, 3) = -7;
    A.emplace(0, 0) = 8;
    A.compress();
    CHECK(A.coeff(0, 0) == 50); // 42 + 8
    CHECK(A.coeff(3, 3) == -7);
    CHECK(A.coeff(1, 1) == 0);
}

TEST_CASE("sparse_float_value_type", "[sparse]") {
    storage::SparseCoordinateAccessor<float, extents<3, 3>> A;
    A.emplace(1, 2) = 3.14f;
    A.compress();
    CHECK(A.coeff(1, 2) == Catch::Approx(3.14f));
    CHECK(A.coeff(0, 0) == 0.0f);
}

TEST_CASE("sparse_copy_semantics", "[sparse]") {
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(2, 2) = 2.0;
    A.compress();

    // Copy construct
    auto B = A;
    CHECK(B.coeff(0, 0) == 1.0);
    CHECK(B.coeff(2, 2) == 2.0);

    // Modify copy -- should not affect original
    B.emplace(1, 1) = 99.0;
    B.compress();
    CHECK(B.coeff(1, 1) == 99.0);
    CHECK(A.coeff(1, 1) == 0.0); // original unchanged
}

TEST_CASE("sparse_move_semantics", "[sparse]") {
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(2, 2) = 2.0;
    A.compress();

    auto B = std::move(A);
    CHECK(B.coeff(0, 0) == 1.0);
    CHECK(B.coeff(2, 2) == 2.0);
}

TEST_CASE("sparse_empty_iterator", "[sparse]") {
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    // Empty and compressed
    REQUIRE(A.is_compressed());
    CHECK(A.begin() == A.end());
    CHECK(A.cbegin() == A.cend());
}

TEST_CASE("sparse_empty_coeff_ref_throws", "[sparse]") {
    // coeff_ref and const_coeff_ref on empty accessor should throw
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    REQUIRE(A.is_compressed());
    CHECK_THROWS_AS(A.coeff_ref(0, 0), std::invalid_argument);
    CHECK_THROWS_AS(A.coeff_ref(1, 2), std::invalid_argument);
    const auto& cA = A;
    CHECK_THROWS_AS(cA.const_coeff_ref(0, 0), std::invalid_argument);
}

TEST_CASE("sparse_compress_idempotent", "[sparse]") {
    // Calling compress() multiple times without emplace should be a no-op
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    A.emplace(1, 1) = 5.0;
    A.compress();
    CHECK(A.coeff(1, 1) == 5.0);

    A.compress(); // no-op
    CHECK(A.coeff(1, 1) == 5.0);

    A.compress(); // still no-op
    CHECK(A.coeff(1, 1) == 5.0);
}

TEST_CASE("sparse_boundary_indices", "[sparse]") {
    // Test first and last valid indices
    storage::SparseCoordinateAccessor<double, extents<5, 5>> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(4, 4) = 2.0;
    A.compress();
    CHECK(A.coeff(0, 0) == 1.0);
    CHECK(A.coeff(4, 4) == 2.0);
    CHECK(A.coeff(0, 4) == 0.0);
    CHECK(A.coeff(4, 0) == 0.0);
}

TEST_CASE("sparse_large_extents_few_nonzeros", "[sparse]") {
    // Large matrix with very few entries -- exercises binary search efficiency
    storage::SparseCoordinateAccessor<double, extents<1000, 1000>> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(999, 999) = 2.0;
    A.emplace(500, 500) = 3.0;
    A.compress();

    CHECK(A.coeff(0, 0) == 1.0);
    CHECK(A.coeff(500, 500) == 3.0);
    CHECK(A.coeff(999, 999) == 2.0);
    CHECK(A.coeff(100, 100) == 0.0);
    CHECK(A.coeff(999, 0) == 0.0);
}

TEST_CASE("sparse_dynamic_nonsquare_extents", "[sparse]") {
    // Non-square dynamic extents
    storage::SparseCoordinateAccessor<double, dextents<2>> A(
        create_dextents(3, 7));
    A.emplace(0, 6) = 1.0;
    A.emplace(2, 0) = 2.0;
    A.emplace(1, 3) = 3.0;
    A.compress();

    CHECK(A.coeff(0, 6) == 1.0);
    CHECK(A.coeff(2, 0) == 2.0);
    CHECK(A.coeff(1, 3) == 3.0);
    CHECK(A.coeff(0, 0) == 0.0);
    CHECK(A.coeff(2, 6) == 0.0);
}

TEST_CASE("sparse_dynamic_rank1", "[sparse]") {
    storage::SparseCoordinateAccessor<double, dextents<1>> A(
        create_dextents(index_type(10)));
    A.emplace(3) = 5.0;
    A.emplace(7) = 9.0;
    A.compress();
    CHECK(A.coeff(0) == 0.0);
    CHECK(A.coeff(3) == 5.0);
    CHECK(A.coeff(7) == 9.0);
    CHECK(A.coeff(9) == 0.0);
}

TEST_CASE("sparse_dynamic_rank3", "[sparse]") {
    storage::SparseCoordinateAccessor<double, dextents<3>> A(
        create_dextents(2, 3, 4));
    A.emplace(1, 2, 3) = 42.0;
    A.emplace(0, 0, 0) = 1.0;
    A.compress();
    CHECK(A.coeff(1, 2, 3) == 42.0);
    CHECK(A.coeff(0, 0, 0) == 1.0);
    CHECK(A.coeff(0, 1, 2) == 0.0);
}

TEST_CASE("sparse_const_coeff_ref_existing", "[sparse]") {
    // const_coeff_ref should return correct reference for existing entries
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    A.emplace(1, 2) = 42.0;
    A.compress();
    const auto& cA = A;
    CHECK(cA.const_coeff_ref(1, 2) == 42.0);
}

TEST_CASE("sparse_coeff_ref_modification", "[sparse]") {
    // Modifying through coeff_ref should persist
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    A.emplace(1, 1) = 10.0;
    A.compress();
    CHECK(A.coeff(1, 1) == 10.0);

    A.coeff_ref(1, 1) = 99.0;
    CHECK(A.coeff(1, 1) == 99.0);
}

TEST_CASE("sparse_iterator_decrement", "[sparse]") {
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(1, 1) = 2.0;
    A.emplace(2, 2) = 3.0;
    A.compress();

    auto it = A.end();
    --it;
    CHECK(it.value() == 3.0); // last element (2,2)
    --it;
    CHECK(it.value() == 2.0); // middle element (1,1)
    --it;
    CHECK(it.value() == 1.0); // first element (0,0)
    CHECK(it == A.begin());
}

TEST_CASE("sparse_iterator_difference", "[sparse]") {
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(1, 1) = 2.0;
    A.emplace(2, 2) = 3.0;
    A.compress();

    CHECK(A.end() - A.begin() == 3);
    CHECK(A.begin() - A.end() == -3);

    auto it = A.begin();
    it += 2;
    CHECK(it.value() == 3.0);
    CHECK(it - A.begin() == 2);
}

// ── index_set tests ─────────────────────────────────────────────────────────

TEST_CASE("sparse_index_set_rank2_col_for_row", "[sparse][index_set]") {
    // Matrix with entries at (0,1), (0,3), (1,2), (2,0), (2,3)
    storage::SparseCoordinateAccessor<double, extents<4, 5>> A;
    A.emplace(0, 1) = 1.0;
    A.emplace(0, 3) = 2.0;
    A.emplace(1, 2) = 3.0;
    A.emplace(2, 0) = 4.0;
    A.emplace(2, 3) = 5.0;
    A.compress();

    SECTION("row 0 has columns {1, 3}") {
        auto cols = A.index_set<1>(0);
        REQUIRE(cols.size() == 2);
        auto it = cols.begin();
        CHECK(*it++ == 1);
        CHECK(*it++ == 3);
        CHECK(cols.contains(1));
        CHECK(cols.contains(3));
        CHECK(!cols.contains(0));
        CHECK(!cols.contains(2));
    }

    SECTION("row 1 has columns {2}") {
        auto cols = A.index_set<1>(1);
        REQUIRE(cols.size() == 1);
        CHECK(*cols.begin() == 2);
    }

    SECTION("row 2 has columns {0, 3}") {
        auto cols = A.index_set<1>(2);
        REQUIRE(cols.size() == 2);
        auto it = cols.begin();
        CHECK(*it++ == 0);
        CHECK(*it++ == 3);
    }

    SECTION("row 3 has no entries") {
        auto cols = A.index_set<1>(3);
        CHECK(cols.size() == 0);
        CHECK(cols.empty());
    }

    SECTION("convenience alias col_range_for_row") {
        auto cols = A.col_range_for_row(0);
        CHECK(cols.size() == 2);
    }
}

TEST_CASE("sparse_index_set_rank2_row_for_col", "[sparse][index_set]") {
    // Same matrix as above: (0,1), (0,3), (1,2), (2,0), (2,3)
    storage::SparseCoordinateAccessor<double, extents<4, 5>> A;
    A.emplace(0, 1) = 1.0;
    A.emplace(0, 3) = 2.0;
    A.emplace(1, 2) = 3.0;
    A.emplace(2, 0) = 4.0;
    A.emplace(2, 3) = 5.0;
    A.compress();

    SECTION("col 0 has rows {2}") {
        auto rows = A.index_set<0>(0);
        REQUIRE(rows.size() == 1);
        CHECK(*rows.begin() == 2);
    }

    SECTION("col 1 has rows {0}") {
        auto rows = A.index_set<0>(1);
        REQUIRE(rows.size() == 1);
        CHECK(*rows.begin() == 0);
    }

    SECTION("col 3 has rows {0, 2}") {
        auto rows = A.index_set<0>(3);
        REQUIRE(rows.size() == 2);
        auto it = rows.begin();
        CHECK(*it++ == 0);
        CHECK(*it++ == 2);
    }

    SECTION("col 4 has no entries") {
        auto rows = A.index_set<0>(4);
        CHECK(rows.size() == 0);
        CHECK(rows.empty());
    }

    SECTION("convenience alias row_range_for_col") {
        auto rows = A.row_range_for_col(3);
        CHECK(rows.size() == 2);
    }
}

TEST_CASE("sparse_index_set_rank1", "[sparse][index_set]") {
    storage::SparseCoordinateAccessor<double, extents<10>> V;
    V.emplace(2) = 1.0;
    V.emplace(5) = 2.0;
    V.emplace(7) = 3.0;
    V.compress();

    auto idx = V.index_set<0>();
    REQUIRE(idx.size() == 3);
    auto it = idx.begin();
    CHECK(*it++ == 2);
    CHECK(*it++ == 5);
    CHECK(*it++ == 7);
    CHECK(idx.contains(2));
    CHECK(idx.contains(5));
    CHECK(idx.contains(7));
    CHECK(!idx.contains(0));
    CHECK(!idx.contains(3));

    SECTION("convenience alias nonzero_segment") {
        auto seg = V.nonzero_segment();
        CHECK(seg.size() == 3);
    }
}

TEST_CASE("sparse_index_set_empty_matrix", "[sparse][index_set]") {
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    // Empty but compressed (default state)
    REQUIRE(A.is_compressed());

    auto cols = A.index_set<1>(0);
    CHECK(cols.size() == 0);
    CHECK(cols.empty());

    auto rows = A.index_set<0>(0);
    CHECK(rows.size() == 0);
    CHECK(rows.empty());
}

TEST_CASE("sparse_index_set_diagonal", "[sparse][index_set]") {
    // Diagonal matrix — each row has exactly one column
    storage::SparseCoordinateAccessor<double, extents<3, 3>> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(1, 1) = 2.0;
    A.emplace(2, 2) = 3.0;
    A.compress();

    for (index_type i = 0; i < 3; ++i) {
        auto cols = A.index_set<1>(i);
        REQUIRE(cols.size() == 1);
        CHECK(*cols.begin() == i);

        auto rows = A.index_set<0>(i);
        REQUIRE(rows.size() == 1);
        CHECK(*rows.begin() == i);
    }
}

TEST_CASE("sparse_index_set_has_index_set_trait", "[sparse][index_set]") {
    using SCA = storage::SparseCoordinateAccessor<double, extents<3, 3>>;
    using traits = expression::detail::ExpressionTraits<SCA>;
    STATIC_REQUIRE(traits::has_index_set);
    STATIC_REQUIRE(traits::has_known_zeros);
    STATIC_REQUIRE(!traits::access_features.is_reference);
}
