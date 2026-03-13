#include "catch_include.hpp"

#include <zipper/Array.hpp>
#include <zipper/Form.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/MDSpan.hpp>
#include <zipper/expression/nullary/StlMDArray.hpp>

// ============================================================================
// STL deduction guide tests
// ============================================================================

// --- VectorBase deduction guides (rank-1 STL containers) ---

TEST_CASE("stl_deduction_vector_rvalue_vector", "[stl_deduction]") {
  // rvalue std::vector → owning StlMDArray<std::vector<double>>
  auto V = zipper::VectorBase(std::vector<double>{1.0, 2.0, 3.0});

  // Should be owning: StlMDArray<std::vector<double>> (no reference)
  static_assert(!std::decay_t<decltype(V)>::stores_references,
                "rvalue STL should produce owning (non-reference) storage");

  CHECK(V(0) == 1.0);
  CHECK(V(1) == 2.0);
  CHECK(V(2) == 3.0);

  // Should be writable
  V(0) = 10.0;
  CHECK(V(0) == 10.0);
}

TEST_CASE("stl_deduction_vector_lvalue_vector", "[stl_deduction]") {
  // lvalue std::vector → borrowing StlMDArray<std::vector<double>&>
  std::vector<double> data = {4.0, 5.0, 6.0};
  auto V = zipper::VectorBase(data);

  static_assert(std::decay_t<decltype(V)>::stores_references,
                "lvalue STL should produce borrowing (reference) storage");

  CHECK(V(0) == 4.0);
  CHECK(V(1) == 5.0);
  CHECK(V(2) == 6.0);

  // Write-through: modifications should be visible in original data
  V(0) = 40.0;
  CHECK(data[0] == 40.0);

  // And vice versa
  data[1] = 50.0;
  CHECK(V(1) == 50.0);
}

TEST_CASE("stl_deduction_vector_rvalue_array", "[stl_deduction]") {
  // rvalue std::array → owning StlMDArray<std::array<double,3>>
  auto V = zipper::VectorBase(std::array<double, 3>{{7.0, 8.0, 9.0}});

  static_assert(!std::decay_t<decltype(V)>::stores_references,
                "rvalue std::array should produce owning storage");

  CHECK(V(0) == 7.0);
  CHECK(V(1) == 8.0);
  CHECK(V(2) == 9.0);
}

TEST_CASE("stl_deduction_vector_lvalue_array", "[stl_deduction]") {
  // lvalue std::array → borrowing StlMDArray<std::array<double,3>&>
  std::array<double, 3> data{{10.0, 11.0, 12.0}};
  auto V = zipper::VectorBase(data);

  static_assert(std::decay_t<decltype(V)>::stores_references,
                "lvalue std::array should produce borrowing storage");

  CHECK(V(0) == 10.0);
  CHECK(V(1) == 11.0);
  CHECK(V(2) == 12.0);

  // Write-through
  V(2) = 120.0;
  CHECK(data[2] == 120.0);
}

// --- MatrixBase deduction guides (rank-2 STL containers) ---

TEST_CASE("stl_deduction_matrix_rvalue", "[stl_deduction]") {
  // rvalue std::vector<std::array<double,3>> → owning, rank 2
  auto M = zipper::MatrixBase(
      std::vector<std::array<double, 3>>{{1, 2, 3}, {4, 5, 6}});

  static_assert(!std::decay_t<decltype(M)>::stores_references,
                "rvalue rank-2 STL should produce owning storage");

  CHECK(M(0, 0) == 1);
  CHECK(M(0, 1) == 2);
  CHECK(M(0, 2) == 3);
  CHECK(M(1, 0) == 4);
  CHECK(M(1, 1) == 5);
  CHECK(M(1, 2) == 6);
}

TEST_CASE("stl_deduction_matrix_lvalue", "[stl_deduction]") {
  // lvalue std::vector<std::array<double,3>> → borrowing, rank 2
  std::vector<std::array<double, 3>> data = {{1, 2, 3}, {4, 5, 6}};
  auto M = zipper::MatrixBase(data);

  static_assert(std::decay_t<decltype(M)>::stores_references,
                "lvalue rank-2 STL should produce borrowing storage");

  CHECK(M(0, 0) == 1);
  CHECK(M(1, 2) == 6);

  // Write-through
  M(0, 0) = 100;
  CHECK(data[0][0] == 100);
}

TEST_CASE("stl_deduction_matrix_array_of_arrays", "[stl_deduction]") {
  // rvalue std::array<std::array<double,3>,2> → owning, rank 2
  auto M = zipper::MatrixBase(
      std::array<std::array<double, 3>, 2>{{{1, 2, 3}, {4, 5, 6}}});

  static_assert(!std::decay_t<decltype(M)>::stores_references);
  CHECK(M(0, 0) == 1);
  CHECK(M(0, 2) == 3);
  CHECK(M(1, 0) == 4);
  CHECK(M(1, 2) == 6);
}

// --- TensorBase deduction guides (any-rank STL containers) ---

TEST_CASE("stl_deduction_tensor_rvalue_rank1", "[stl_deduction]") {
  auto T = zipper::TensorBase(std::vector<double>{1.0, 2.0, 3.0});

  static_assert(!std::decay_t<decltype(T)>::stores_references);
  CHECK(T(0) == 1.0);
  CHECK(T(1) == 2.0);
  CHECK(T(2) == 3.0);
}

TEST_CASE("stl_deduction_tensor_lvalue_rank2", "[stl_deduction]") {
  std::vector<std::array<double, 2>> data = {{1, 2}, {3, 4}, {5, 6}};
  auto T = zipper::TensorBase(data);

  static_assert(std::decay_t<decltype(T)>::stores_references);
  CHECK(T(0, 0) == 1);
  CHECK(T(2, 1) == 6);

  // Write-through
  T(1, 0) = 30;
  CHECK(data[1][0] == 30);
}

// --- ArrayBase deduction guides (any-rank STL containers) ---

TEST_CASE("stl_deduction_array_rvalue", "[stl_deduction]") {
  auto A = zipper::ArrayBase(std::array<double, 4>{{10, 20, 30, 40}});

  static_assert(!std::decay_t<decltype(A)>::stores_references);
  CHECK(A(0) == 10);
  CHECK(A(3) == 40);
}

TEST_CASE("stl_deduction_array_lvalue", "[stl_deduction]") {
  std::array<double, 4> data{{10, 20, 30, 40}};
  auto A = zipper::ArrayBase(data);

  static_assert(std::decay_t<decltype(A)>::stores_references);
  CHECK(A(0) == 10);
  CHECK(A(3) == 40);

  A(0) = 100;
  CHECK(data[0] == 100);
}

// --- FormBase deduction guides (any-rank STL containers) ---

TEST_CASE("stl_deduction_form_rvalue", "[stl_deduction]") {
  auto F = zipper::FormBase(std::vector<double>{2.0, 4.0, 6.0});

  static_assert(!std::decay_t<decltype(F)>::stores_references);
  CHECK(F(0) == 2.0);
  CHECK(F(1) == 4.0);
  CHECK(F(2) == 6.0);
}

TEST_CASE("stl_deduction_form_lvalue", "[stl_deduction]") {
  std::vector<double> data = {2.0, 4.0, 6.0};
  auto F = zipper::FormBase(data);

  static_assert(std::decay_t<decltype(F)>::stores_references);
  CHECK(F(0) == 2.0);

  F(1) = 40.0;
  CHECK(data[1] == 40.0);
}

// --- stores_references trait correctness ---

TEST_CASE("stl_deduction_stores_references_trait", "[stl_deduction]") {
  // Owning (rvalue) → stores_references == false
  {
    using OwningVec = zipper::VectorBase<
        zipper::expression::nullary::StlMDArray<std::vector<double>>>;
    static_assert(!OwningVec::stores_references);
  }
  // Borrowing (lvalue ref) → stores_references == true
  {
    using BorrowVec = zipper::VectorBase<
        zipper::expression::nullary::StlMDArray<std::vector<double> &>>;
    static_assert(BorrowVec::stores_references);
  }
  // Const borrowing → stores_references == true
  {
    using ConstBorrowVec = zipper::VectorBase<
        zipper::expression::nullary::StlMDArray<const std::vector<double> &>>;
    static_assert(ConstBorrowVec::stores_references);
  }
}

// ============================================================================
// Original MDSpan-based tests
// ============================================================================

TEST_CASE("stl_storage_zipper_bases", "[data]") {
  // Vector span wrapping a std::vector via MDSpan (dynamic extent)
  {
    std::vector<double> x = {0, 1, 2};
    using SpanExpr =
        zipper::expression::nullary::MDSpan<double, zipper::extents<zipper::dynamic_extent>>;
    using VSpan = zipper::VectorBase<SpanExpr>;
    VSpan X{std::span<double>(x), zipper::extents<zipper::dynamic_extent>(x.size())};

    CHECK(X(0) == 0);
    CHECK(X(1) == 1);
    CHECK(X(2) == 2);

    CHECK(X.expression().coeff(0) == 0);
    CHECK(X.expression().coeff(1) == 1);
    CHECK(X.expression().coeff(2) == 2);
    CHECK(X.expression().coeff_ref(0) == 0);
    CHECK(X.expression().coeff_ref(1) == 1);
    CHECK(X.expression().coeff_ref(2) == 2);
    CHECK(X.expression().const_coeff_ref(0) == 0);
    CHECK(X.expression().const_coeff_ref(1) == 1);
    CHECK(X.expression().const_coeff_ref(2) == 2);

    X(0) = 4;
    X(1) = 5;
    X(2) = 6;
    CHECK(X(0) == 4);
    CHECK(X(1) == 5);
    CHECK(X(2) == 6);

    using expression_type = std::decay_t<decltype(X)>::expression_type;
    using traits =
        zipper::expression::detail::ExpressionTraits<expression_type>;

    static_assert(!traits::access_features.is_const);
    static_assert(traits::is_writable);
  }

  // Vector span wrapping a std::array via span_type (static extent)
  {
    std::array<double, 3> x{{0, 1, 2}};
    auto sp = std::span<double, 3>(x);
    zipper::Vector<double, 3>::span_type X{sp};

    CHECK(X(0) == 0);
    CHECK(X(1) == 1);
    CHECK(X(2) == 2);

    CHECK(X.expression().coeff(0) == 0);
    CHECK(X.expression().coeff(1) == 1);
    CHECK(X.expression().coeff(2) == 2);
    CHECK(X.expression().coeff_ref(0) == 0);
    CHECK(X.expression().coeff_ref(1) == 1);
    CHECK(X.expression().coeff_ref(2) == 2);
    CHECK(X.expression().const_coeff_ref(0) == 0);
    CHECK(X.expression().const_coeff_ref(1) == 1);
    CHECK(X.expression().const_coeff_ref(2) == 2);

    X(0) = 4;
    X(1) = 5;
    X(2) = 6;

    CHECK(X(0) == 4);
    CHECK(X(1) == 5);
    CHECK(X(2) == 6);

    // Verify data is written through to underlying std::array
    CHECK(x[0] == 4);
    CHECK(x[1] == 5);
    CHECK(x[2] == 6);

    X = {0, 4, 5};
    CHECK(X.size() == 3);
    CHECK(X.rows() == 3);

    CHECK(X(0) == 0);
    CHECK(X(1) == 4);
    CHECK(X(2) == 5);
  }

  // Matrix span wrapping a flat std::array (2x3)
  {
    std::array<double, 6> x{{0, 1, 2, 3, 4, 5}};
    auto sp = std::span<double, 6>(x);
    zipper::Matrix<double, 2, 3>::span_type X{sp};

    CHECK(X(0, 0) == 0);
    CHECK(X(0, 1) == 1);
    CHECK(X(0, 2) == 2);
    CHECK(X(1, 0) == 3);
    CHECK(X(1, 1) == 4);
    CHECK(X(1, 2) == 5);

    CHECK(X.expression().coeff(0, 0) == 0);
    CHECK(X.expression().coeff(0, 1) == 1);
    CHECK(X.expression().coeff(0, 2) == 2);
    CHECK(X.expression().coeff(1, 0) == 3);
    CHECK(X.expression().coeff(1, 1) == 4);
    CHECK(X.expression().coeff(1, 2) == 5);
    CHECK(X.expression().coeff_ref(0, 0) == 0);
    CHECK(X.expression().coeff_ref(0, 1) == 1);
    CHECK(X.expression().coeff_ref(0, 2) == 2);
    CHECK(X.expression().coeff_ref(1, 0) == 3);
    CHECK(X.expression().coeff_ref(1, 1) == 4);
    CHECK(X.expression().coeff_ref(1, 2) == 5);
    CHECK(X.expression().const_coeff_ref(0, 0) == 0);
    CHECK(X.expression().const_coeff_ref(0, 1) == 1);
    CHECK(X.expression().const_coeff_ref(0, 2) == 2);
    CHECK(X.expression().const_coeff_ref(1, 0) == 3);
    CHECK(X.expression().const_coeff_ref(1, 1) == 4);
    CHECK(X.expression().const_coeff_ref(1, 2) == 5);

    X(0, 0) = 14;
    X(0, 1) = 15;
    X(0, 2) = 16;
    X(1, 0) = 24;
    X(1, 1) = 25;
    X(1, 2) = 26;

    CHECK(x[0] == 14);
    CHECK(x[1] == 15);
    CHECK(x[2] == 16);
    CHECK(x[3] == 24);
    CHECK(x[4] == 25);
    CHECK(x[5] == 26);

    CHECK(X(0, 0) == 14);
    CHECK(X(0, 1) == 15);
    CHECK(X(0, 2) == 16);
    CHECK(X(1, 0) == 24);
    CHECK(X(1, 1) == 25);
    CHECK(X(1, 2) == 26);

    zipper::Matrix<double, 2, 3> B;
    CHECK(B.extents() == zipper::create_dextents(2, 3));
    CHECK(X.extents() == zipper::create_dextents(2, 3));
    B.row(0) = {0, 1, 2};
    B.row(1) = {3, 4, 5};
    X = B;
    CHECK(X.extents() == zipper::create_dextents(2, 3));
    CHECK((X == B));
  }
}
