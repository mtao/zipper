

#include <iostream>
#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/MatrixBase.hxx>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/Random.hpp>
#include <zipper/expression/reductions/Contraction.hpp>

#include "../../catch_include.hpp"

// ── contract<I,J>() free function ───────────────────────────────────

TEST_CASE("contract replaces manual PartialTrace",
          "[tensor][contraction]") {
    // Matrix product via tensor product + contract<1,2>
    // should match MatrixBase operator*
    zipper::Tensor<double, 3, 3> I =
        zipper::expression::nullary::Identity<double>{};
    zipper::Tensor<double, 3, 3> M;
    M = zipper::expression::nullary::normal_random_infinite<double>(0, 1);

    // contract<1,2> on rank-4 tensor product → rank-2 (matrix product)
    auto TP = I * M;
    STATIC_CHECK(decltype(TP)::extents_type::rank() == 4);

    auto contracted = zipper::contract<1, 2>(TP);
    STATIC_CHECK(decltype(contracted)::extents_type::rank() == 2);

    // Should equal I * M = M
    zipper::Tensor<double, 3, 3> result = contracted;
    for (zipper::index_type j = 0; j < 3; ++j) {
        for (zipper::index_type k = 0; k < 3; ++k) {
            CHECK(result(j, k) == Catch::Approx(M(j, k)));
        }
    }
}

TEST_CASE("contract on rank-2 gives trace",
          "[tensor][contraction]") {
    zipper::Tensor<double, 3, 3> I =
        zipper::expression::nullary::Identity<double>{};

    // contract<0,1> on a rank-2 identity should give a scalar rank-0
    // via PartialTrace.  But PartialTrace<..., 0, 1> on rank-2 gives rank-0.
    // Actually PartialTrace returns an expression with extents_type::rank() == 0,
    // which is just a scalar accessor.  The TensorBase wrapper around it
    // should still work.
    auto contracted = zipper::contract<0, 1>(I);
    STATIC_CHECK(decltype(contracted)::extents_type::rank() == 0);

    // The trace of a 3x3 identity is 3
    CHECK(contracted() == Catch::Approx(3.0));
}

TEST_CASE("contract general matrix trace",
          "[tensor][contraction]") {
    zipper::Tensor<double, 3, 3> M;
    M(0, 0) = 1;
    M(1, 1) = 2;
    M(2, 2) = 3;
    M(0, 1) = 7;
    M(1, 0) = 8;
    M(0, 2) = 9;
    M(2, 0) = 10;
    M(1, 2) = 11;
    M(2, 1) = 12;

    auto trace_tensor = zipper::contract<0, 1>(M);
    // Trace = 1 + 2 + 3 = 6
    CHECK(trace_tensor() == Catch::Approx(6.0));
}

// ── tensor_product() named free function ────────────────────────────

TEST_CASE("tensor_product matches operator*",
          "[tensor][tensor_product]") {
    zipper::Tensor<double, 2, 2> A;
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;

    zipper::Tensor<double, 2> v;
    v(0) = 5;
    v(1) = 6;

    auto tp_named = zipper::tensor_product(A, v);
    auto tp_op = A * v;

    STATIC_CHECK(decltype(tp_named)::extents_type::rank() == 3);
    STATIC_CHECK(decltype(tp_op)::extents_type::rank() == 3);

    for (zipper::index_type i = 0; i < 2; ++i) {
        for (zipper::index_type j = 0; j < 2; ++j) {
            for (zipper::index_type k = 0; k < 2; ++k) {
                CHECK(tp_named(i, j, k) ==
                      Catch::Approx(tp_op(i, j, k)));
                // Should be A(i,j) * v(k)
                CHECK(tp_named(i, j, k) ==
                      Catch::Approx(A(i, j) * v(k)));
            }
        }
    }
}

// ── full_contract() ─────────────────────────────────────────────────

TEST_CASE("full_contract on rank-2 is matrix trace",
          "[tensor][contraction]") {
    zipper::Tensor<double, 3, 3> I =
        zipper::expression::nullary::Identity<double>{};

    double result = zipper::full_contract(I);
    CHECK(result == Catch::Approx(3.0));
}

TEST_CASE("full_contract on rank-4 identity tensor",
          "[tensor][contraction]") {
    // Build a rank-4 tensor T where T(i,j,k,l) = delta(i,l)*delta(j,k)
    // full_contract folds in half: sum_{i,j} T(i,j,j,i) = sum_{i,j} delta(i,i)*delta(j,j) = n*n
    zipper::Tensor<double, 3, 3> I =
        zipper::expression::nullary::Identity<double>{};

    // I * I gives rank-4: (I*I)(i,j,k,l) = I(i,j) * I(k,l) = delta(i,j)*delta(k,l)
    auto T = I * I;
    STATIC_CHECK(decltype(T)::extents_type::rank() == 4);

    // full_contract sums T(i,j,j,i) = delta(i,j)*delta(j,i)
    // = delta(i,j)*delta(i,j)
    // = sum_i delta(i,i) = 3
    double result = zipper::full_contract(T);
    CHECK(result == Catch::Approx(3.0));
}

TEST_CASE("full_contract custom rank-2 tensor",
          "[tensor][contraction]") {
    zipper::Tensor<double, 2, 2> M;
    M(0, 0) = 10;
    M(0, 1) = 20;
    M(1, 0) = 30;
    M(1, 1) = 40;

    // full_contract on rank-2: sum_i M(i, i) = M(0,0) + M(1,1) = 50
    double result = zipper::full_contract(M);
    CHECK(result == Catch::Approx(50.0));
}

TEST_CASE("full_contract rank-4 custom",
          "[tensor][contraction]") {
    // Build a rank-4 tensor from two rank-2 tensors
    zipper::Tensor<double, 2, 2> A;
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;

    zipper::Tensor<double, 2, 2> B;
    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;

    auto T = A * B;  // rank-4: T(i,j,k,l) = A(i,j) * B(k,l)

    // full_contract: sum_{i,j} T(i,j,j,i)
    // = sum_{i,j} A(i,j) * B(j,i)
    // = A(0,0)*B(0,0) + A(0,1)*B(1,0) + A(1,0)*B(0,1) + A(1,1)*B(1,1)
    // = 1*5 + 2*7 + 3*6 + 4*8
    // = 5 + 14 + 18 + 32
    // = 69
    double result = zipper::full_contract(T);
    CHECK(result == Catch::Approx(69.0));
}

// ── contract + tensor_product pipeline ──────────────────────────────

TEST_CASE("tensor_product then contract gives matrix product",
          "[tensor][contraction][integration]") {
    zipper::Tensor<double, 2, 3> A;
    zipper::Tensor<double, 3, 4> B;

    // Fill with known values
    for (zipper::index_type i = 0; i < 2; ++i)
        for (zipper::index_type j = 0; j < 3; ++j)
            A(i, j) = static_cast<double>(i * 3 + j + 1);

    for (zipper::index_type i = 0; i < 3; ++i)
        for (zipper::index_type j = 0; j < 4; ++j)
            B(i, j) = static_cast<double>(i * 4 + j + 1);

    // tensor_product gives rank-4: (A*B)(i,j,k,l) = A(i,j) * B(k,l)
    auto TP = zipper::tensor_product(A, B);
    STATIC_CHECK(decltype(TP)::extents_type::rank() == 4);

    // contract<1,2> traces over j,k (the inner dimensions) → rank-2 result
    // result(i,l) = sum_j A(i,j) * B(j,l)  → matrix product
    auto result = zipper::contract<1, 2>(TP);
    STATIC_CHECK(decltype(result)::extents_type::rank() == 2);

    // Compute matrix product manually
    zipper::Matrix<double, 2, 4> expected;
    for (zipper::index_type i = 0; i < 2; ++i) {
        for (zipper::index_type l = 0; l < 4; ++l) {
            double sum = 0;
            for (zipper::index_type j = 0; j < 3; ++j) {
                sum += A(i, j) * B(j, l);
            }
            expected(i, l) = sum;
        }
    }

    zipper::Tensor<double, 2, 4> eval_result = result;
    for (zipper::index_type i = 0; i < 2; ++i) {
        for (zipper::index_type l = 0; l < 4; ++l) {
            CHECK(eval_result(i, l) == Catch::Approx(expected(i, l)));
        }
    }
}
