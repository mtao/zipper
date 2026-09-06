#include <type_traits>

#include <zipper/COOMatrix.hpp>
#include <zipper/COOVector.hpp>
#include <zipper/CSRMatrix.hpp>
#include <zipper/CSRVector.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>

#include "catch_include.hpp"

namespace {

template <typename E>
concept DeducesSparseMatrix = requires(const E &expr) {
    zipper::COOMatrix{expr};
    zipper::CSMatrix{expr};
    zipper::CSRMatrix{expr};
};

template <typename E>
concept DeducesSparseVector = requires(const E &expr) {
    zipper::COOVector{expr};
    zipper::CSVector{expr};
    zipper::CSRVector{expr};
};

} // namespace

TEST_CASE("sparse_matrix_raw_expression_deduction", "[sparse][ctad][matrix]") {
    auto check = []<zipper::index_type R, zipper::index_type C>() {
        zipper::Matrix<double, R, C> source{{1.0, 0.0, 2.0},
                                           {0.0, 3.0, 0.0}};
        auto sum = source + source;
        const auto &raw = sum.expression();
        static_assert(DeducesSparseMatrix<std::decay_t<decltype(raw)>>);
        static_assert(!DeducesSparseVector<std::decay_t<decltype(raw)>>);

        zipper::COOMatrix coo(raw);
        zipper::CSMatrix cs(raw);
        zipper::CSRMatrix csr(raw);
        static_assert(std::is_same_v<decltype(coo), zipper::COOMatrix<double, R, C>>);
        static_assert(std::is_same_v<decltype(cs), zipper::CSMatrix<double, R, C>>);
        static_assert(std::is_same_v<decltype(csr), zipper::CSRMatrix<double, R, C>>);
        static_assert(std::is_same_v<typename decltype(cs)::layout_policy,
                                     zipper::storage::default_layout_policy>);

        source(0, 0) = 100.0;
        auto check_values = [](const auto &owner) {
            CHECK(owner.extent(0) == 2);
            CHECK(owner.extent(1) == 3);
            CHECK(owner(0, 0) == 2.0);
            CHECK(owner(0, 1) == 0.0);
            CHECK(owner(0, 2) == 4.0);
            CHECK(owner(1, 0) == 0.0);
            CHECK(owner(1, 1) == 6.0);
            CHECK(owner(1, 2) == 0.0);
        };
        check_values(coo);
        check_values(cs);
        check_values(csr);

        auto view = cs.as_const_span();
        static_assert(std::is_const_v<
                      typename decltype(view)::expression_type::value_type>);
        zipper::COOMatrix coo_copy(view.expression());
        zipper::CSMatrix cs_copy(view.expression());
        zipper::CSRMatrix csr_copy(view.expression());
        static_assert(std::is_same_v<decltype(coo_copy), decltype(coo)>);
        static_assert(std::is_same_v<decltype(cs_copy), decltype(cs)>);
        static_assert(std::is_same_v<decltype(csr_copy), decltype(csr)>);

        cs.coeff_ref(0, 0) = 200.0;
        check_values(coo_copy);
        check_values(cs_copy);
        check_values(csr_copy);
        coo_copy.coeff_ref(0, 0) = 7.0;
        cs_copy.coeff_ref(0, 0) = 8.0;
        csr_copy.coeff_ref(0, 0) = 9.0;
        CHECK(coo_copy(0, 0) == 7.0);
        CHECK(cs_copy(0, 0) == 8.0);
        CHECK(csr_copy(0, 0) == 9.0);
        CHECK(cs(0, 0) == 200.0);
    };

    check.template operator()<2, 3>();
    check.template operator()<zipper::dynamic_extent, zipper::dynamic_extent>();
    check.template operator()<2, zipper::dynamic_extent>();
    check.template operator()<zipper::dynamic_extent, 3>();
}

TEST_CASE("sparse_vector_raw_expression_deduction", "[sparse][ctad][vector]") {
    auto check = []<zipper::index_type N>() {
        zipper::Vector<double, N> source{1.0, 0.0, 3.0};
        auto sum = source + source;
        const auto &raw = sum.expression();
        static_assert(DeducesSparseVector<std::decay_t<decltype(raw)>>);
        static_assert(!DeducesSparseMatrix<std::decay_t<decltype(raw)>>);

        zipper::COOVector coo(raw);
        zipper::CSVector cs(raw);
        zipper::CSRVector csr(raw);
        static_assert(std::is_same_v<decltype(coo), zipper::COOVector<double, N>>);
        static_assert(std::is_same_v<decltype(cs), zipper::CSVector<double, N>>);
        static_assert(std::is_same_v<decltype(csr), zipper::CSRVector<double, N>>);

        source(0) = 100.0;
        auto check_values = [](const auto &owner) {
            CHECK(owner.extent(0) == 3);
            CHECK(owner(0) == 2.0);
            CHECK(owner(1) == 0.0);
            CHECK(owner(2) == 6.0);
        };
        check_values(coo);
        check_values(cs);
        check_values(csr);

        auto view = cs.as_const_span();
        static_assert(std::is_const_v<
                      typename decltype(view)::expression_type::value_type>);
        zipper::COOVector coo_copy(view.expression());
        zipper::CSVector cs_copy(view.expression());
        zipper::CSRVector csr_copy(view.expression());
        static_assert(std::is_same_v<decltype(coo_copy), decltype(coo)>);
        static_assert(std::is_same_v<decltype(cs_copy), decltype(cs)>);
        static_assert(std::is_same_v<decltype(csr_copy), decltype(csr)>);

        cs.coeff_ref(0) = 200.0;
        check_values(coo_copy);
        check_values(cs_copy);
        check_values(csr_copy);
        coo_copy.coeff_ref(0) = 7.0;
        cs_copy.coeff_ref(0) = 8.0;
        csr_copy.coeff_ref(0) = 9.0;
        CHECK(coo_copy(0) == 7.0);
        CHECK(cs_copy(0) == 8.0);
        CHECK(csr_copy(0) == 9.0);
        CHECK(cs(0) == 200.0);
    };

    check.template operator()<3>();
    check.template operator()<zipper::dynamic_extent>();
}

TEST_CASE("sparse_accessor_deduction_preserves_specialized_guides", "[sparse][ctad]") {
    zipper::COOMatrix<double, 2, 3> matrix;
    matrix.emplace(1, 2) = 5.0;
    matrix.compress();
    zipper::COOMatrix matrix_copy(matrix.expression());
    zipper::CSMatrix compressed(matrix);
    zipper::CSMatrix compressed_raw(matrix.expression());
    static_assert(std::is_same_v<decltype(matrix_copy), decltype(matrix)>);
    static_assert(std::is_same_v<decltype(compressed), zipper::CSMatrix<double, 2, 3>>);
    static_assert(std::is_same_v<decltype(compressed_raw), decltype(compressed)>);
    CHECK(matrix_copy(1, 2) == 5.0);
    CHECK(compressed(1, 2) == 5.0);
    CHECK(compressed_raw(1, 2) == 5.0);

    auto csc = matrix.to_csc();
    zipper::CSMatrix csc_copy(csc.expression());
    zipper::CSMatrix cs_copy(compressed.expression());
    zipper::CSRMatrix csr_copy(compressed.expression());
    static_assert(std::is_same_v<decltype(csc_copy), decltype(csc)>);
    static_assert(std::is_same_v<typename decltype(csc_copy)::layout_policy,
                                 zipper::storage::layout_left>);
    static_assert(std::is_same_v<decltype(cs_copy), decltype(compressed)>);
    static_assert(std::is_same_v<decltype(csr_copy), zipper::CSRMatrix<double, 2, 3>>);
    csc.coeff_ref(1, 2) = 10.0;
    CHECK(csc_copy(1, 2) == 5.0);
    CHECK(cs_copy(1, 2) == 5.0);
    CHECK(csr_copy(1, 2) == 5.0);

    zipper::COOVector<double, 3> vector;
    vector.emplace(2) = 7.0;
    vector.compress();
    zipper::COOVector vector_copy(vector.expression());
    zipper::CSVector compressed_vector(vector);
    zipper::CSVector compressed_vector_raw(vector.expression());
    zipper::CSVector cs_vector_copy(compressed_vector.expression());
    zipper::CSRVector csr_vector_copy(compressed_vector.expression());
    static_assert(std::is_same_v<decltype(vector_copy), decltype(vector)>);
    static_assert(std::is_same_v<decltype(compressed_vector),
                                 zipper::CSVector<double, 3>>);
    static_assert(std::is_same_v<decltype(compressed_vector_raw),
                                 decltype(compressed_vector)>);
    static_assert(std::is_same_v<decltype(cs_vector_copy),
                                 decltype(compressed_vector)>);
    static_assert(std::is_same_v<decltype(csr_vector_copy),
                                 zipper::CSRVector<double, 3>>);
    CHECK(vector_copy(2) == 7.0);
    CHECK(compressed_vector(2) == 7.0);
    CHECK(compressed_vector_raw(2) == 7.0);
    CHECK(cs_vector_copy(2) == 7.0);
    CHECK(csr_vector_copy(2) == 7.0);
}
