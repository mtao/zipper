#include <zipper/Array.hpp>
#include <zipper/Form.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/concepts/Algebraic.hpp>
#include <zipper/concepts/DirectSolver.hpp>
#include <zipper/concepts/InnerProductSpace.hpp>
#include <zipper/concepts/Module.hpp>
#include <zipper/concepts/Preconditioner.hpp>
#include <zipper/concepts/Zipper.hpp>
#include <zipper/expression/unary/TriangularView.hpp>
#include <zipper/utils/decomposition/ldlt.hpp>
#include <zipper/utils/decomposition/llt.hpp>
#include <zipper/utils/decomposition/lu.hpp>
#include <zipper/utils/decomposition/qr.hpp>
#include <zipper/utils/solver/preconditioner/jacobi_preconditioner.hpp>
#include <zipper/utils/solver/preconditioner/ssor_preconditioner.hpp>

#include "../catch_include.hpp"

// -----------------------------------------------------------------------
// Type aliases used across multiple test cases
// -----------------------------------------------------------------------

using Tensor3 = zipper::Tensor_<double, zipper::extents<2, 3, 4>>;
using Array2 = zipper::Array<double, 3, 4>;
using Form1 = zipper::Form<double, 3>;

using VBase = zipper::VectorBase<
    zipper::expression::nullary::MDArray<double, zipper::extents<3>>>;
using MBase = zipper::MatrixBase<
    zipper::expression::nullary::MDArray<double, zipper::extents<3, 3>>>;
using TBase = zipper::TensorBase<
    zipper::expression::nullary::MDArray<double, zipper::extents<2, 3, 4>>>;
using ABase = zipper::ArrayBase<
    zipper::expression::nullary::MDArray<double, zipper::extents<3, 4>>>;
using FBase = zipper::FormBase<
    zipper::expression::nullary::MDArray<double, zipper::extents<3>>>;

using MDArr33 =
    zipper::expression::nullary::MDArray<double, zipper::extents<3, 3>>;
using LowerTriView =
    zipper::expression::unary::TriangularView<
        zipper::expression::TriangularMode::Lower, const MDArr33 &>;
using UpperTriView =
    zipper::expression::unary::TriangularView<
        zipper::expression::TriangularMode::Upper, const MDArr33 &>;
using UnitLowerTriView =
    zipper::expression::unary::TriangularView<
        zipper::expression::TriangularMode::UnitLower, const MDArr33 &>;

using LLT3 = zipper::utils::decomposition::LLTResult<double, 3>;
using LDLT3 = zipper::utils::decomposition::LDLTResult<double, 3>;
using QRReduced33 = zipper::utils::decomposition::QRReducedResult<double, 3, 3>;
using QRFull33 = zipper::utils::decomposition::QRFullResult<double, 3, 3>;
using PLU3 = zipper::utils::decomposition::PLUResult<double, 3>;
using PLUX = zipper::utils::decomposition::PLUResult<double, zipper::dynamic_extent>;

using JacobiP3 = zipper::utils::solver::JacobiPreconditioner<double, 3>;
using JacobiPX = zipper::utils::solver::JacobiPreconditioner<double, zipper::dynamic_extent>;
using SSORP3 = zipper::utils::solver::SSORPreconditioner<double, 3>;
using SSORPX = zipper::utils::solver::SSORPreconditioner<double, zipper::dynamic_extent>;

// -----------------------------------------------------------------------
// Concept-constrained helper functions for runtime tests
// -----------------------------------------------------------------------

namespace {

template <zipper::concepts::Zipper Z>
auto zipper_rank(const Z&) -> zipper::rank_type {
    return Z::extents_type::rank();
}

template <zipper::concepts::Vector V>
auto vector_size(const V& v) -> zipper::index_type {
    return v.extent(0);
}

template <zipper::concepts::Matrix M>
auto matrix_rows(const M& m) -> zipper::index_type {
    return m.extent(0);
}

template <zipper::concepts::module V>
auto module_add(const V& a, const V& b) {
    return (a + b).eval();
}

template <zipper::concepts::inner_product_space V>
auto ip_dot(const V& a, const V& b) {
    return a.dot(b);
}

template <zipper::concepts::normed_space V>
auto ns_norm(const V& v) {
    return v.norm();
}

} // namespace

// -----------------------------------------------------------------------
// Zipper wrapper concept: concrete types satisfy their specific concepts
// -----------------------------------------------------------------------

TEST_CASE("zipper_concept_vector", "[concepts]") {
    STATIC_CHECK(zipper::concepts::Vector<zipper::Vector<double, 3>>);
    STATIC_CHECK(zipper::concepts::Zipper<zipper::Vector<double, 3>>);
    STATIC_CHECK(zipper::concepts::Zipper<const zipper::Vector<double, 3>>);
    STATIC_CHECK(zipper::concepts::Zipper<zipper::Vector<double, 3>&>);
    STATIC_CHECK(zipper::concepts::Zipper<const zipper::Vector<double, 3>&>);

    zipper::Vector<double, 3> v;
    v(0) = 1.0;
    v(1) = 2.0;
    v(2) = 3.0;

    REQUIRE(zipper_rank(v) == 1);
    REQUIRE(vector_size(v) == 3);
}

TEST_CASE("zipper_concept_matrix", "[concepts]") {
    STATIC_CHECK(zipper::concepts::Matrix<zipper::Matrix<double, 3, 3>>);
    STATIC_CHECK(zipper::concepts::Zipper<zipper::Matrix<double, 3, 3>>);
    STATIC_CHECK(zipper::concepts::Zipper<const zipper::Matrix<double, 3, 3>>);
    STATIC_CHECK(zipper::concepts::Zipper<zipper::Matrix<double, 3, 3>&>);
    STATIC_CHECK(zipper::concepts::Zipper<const zipper::Matrix<double, 3, 3>&>);

    zipper::Matrix<double, 2, 3> m;
    for (zipper::index_type r = 0; r < 2; ++r)
        for (zipper::index_type c = 0; c < 3; ++c)
            m(r, c) = static_cast<double>(r * 3 + c);

    REQUIRE(zipper_rank(m) == 2);
    REQUIRE(matrix_rows(m) == 2);
}

TEST_CASE("zipper_concept_tensor", "[concepts]") {
    STATIC_CHECK(zipper::concepts::Tensor<Tensor3>);
    STATIC_CHECK(zipper::concepts::Zipper<Tensor3>);
    STATIC_CHECK(zipper::concepts::Zipper<const Tensor3>);
    STATIC_CHECK(zipper::concepts::Zipper<Tensor3&>);
    STATIC_CHECK(zipper::concepts::Zipper<const Tensor3&>);

    Tensor3 t;
    REQUIRE(zipper_rank(t) == 3);
}

TEST_CASE("zipper_concept_array", "[concepts]") {
    STATIC_CHECK(zipper::concepts::Array<Array2>);
    STATIC_CHECK(zipper::concepts::Zipper<Array2>);
    STATIC_CHECK(zipper::concepts::Zipper<const Array2>);
    STATIC_CHECK(zipper::concepts::Zipper<Array2&>);
    STATIC_CHECK(zipper::concepts::Zipper<const Array2&>);

    Array2 a;
    REQUIRE(zipper_rank(a) == 2);
}

TEST_CASE("zipper_concept_form", "[concepts]") {
    STATIC_CHECK(zipper::concepts::Form<Form1>);
    STATIC_CHECK(zipper::concepts::Zipper<Form1>);
    STATIC_CHECK(zipper::concepts::Zipper<const Form1>);
    STATIC_CHECK(zipper::concepts::Zipper<Form1&>);
    STATIC_CHECK(zipper::concepts::Zipper<const Form1&>);

    Form1 f;
    REQUIRE(zipper_rank(f) == 1);
}

// -----------------------------------------------------------------------
// Base types also satisfy Zipper
// -----------------------------------------------------------------------

TEST_CASE("zipper_concept_base_types", "[concepts]") {
    STATIC_CHECK(zipper::concepts::Vector<VBase>);
    STATIC_CHECK(zipper::concepts::Zipper<VBase>);

    STATIC_CHECK(zipper::concepts::Matrix<MBase>);
    STATIC_CHECK(zipper::concepts::Zipper<MBase>);

    STATIC_CHECK(zipper::concepts::Tensor<TBase>);
    STATIC_CHECK(zipper::concepts::Zipper<TBase>);

    STATIC_CHECK(zipper::concepts::Array<ABase>);
    STATIC_CHECK(zipper::concepts::Zipper<ABase>);

    STATIC_CHECK(zipper::concepts::Form<FBase>);
    STATIC_CHECK(zipper::concepts::Zipper<FBase>);
}

// -----------------------------------------------------------------------
// Negative: raw expressions and scalars are NOT Zipper
// -----------------------------------------------------------------------

TEST_CASE("zipper_concept_negative", "[concepts]") {
    STATIC_CHECK_FALSE(zipper::concepts::Zipper<int>);
    STATIC_CHECK_FALSE(zipper::concepts::Zipper<double>);
    STATIC_CHECK_FALSE(zipper::concepts::Zipper<
                       zipper::expression::nullary::MDArray<double, zipper::extents<3>>>);

    // Concepts are mutually exclusive where expected
    STATIC_CHECK_FALSE(zipper::concepts::Matrix<zipper::Vector<double, 3>>);
    STATIC_CHECK_FALSE(zipper::concepts::Vector<zipper::Matrix<double, 3, 3>>);
}

// -----------------------------------------------------------------------
// DirectSolver concept
// -----------------------------------------------------------------------

TEST_CASE("direct_solver_concept", "[concepts]") {
    STATIC_CHECK(zipper::concepts::DirectSolver<LLT3>);
    STATIC_CHECK(zipper::concepts::DirectSolver<LDLT3>);
    STATIC_CHECK(zipper::concepts::DirectSolver<QRReduced33>);
    STATIC_CHECK(zipper::concepts::DirectSolver<QRFull33>);
    STATIC_CHECK(zipper::concepts::DirectSolver<PLU3>);
    STATIC_CHECK(zipper::concepts::DirectSolver<PLUX>);

    STATIC_CHECK(zipper::concepts::DirectSolver<LowerTriView>);
    STATIC_CHECK(zipper::concepts::DirectSolver<UpperTriView>);
    STATIC_CHECK(zipper::concepts::DirectSolver<UnitLowerTriView>);

    STATIC_CHECK_FALSE(zipper::concepts::DirectSolver<int>);
    STATIC_CHECK_FALSE(zipper::concepts::DirectSolver<double>);
    STATIC_CHECK_FALSE(zipper::concepts::DirectSolver<zipper::Matrix<double, 3, 3>>);
}

// -----------------------------------------------------------------------
// SuiteSparse result types satisfy DirectSolver (when available)
// -----------------------------------------------------------------------
#ifdef ZIPPER_HAS_SUITESPARSE
#include <zipper/utils/suitesparse/cholmod.hpp>
#include <zipper/utils/suitesparse/umfpack.hpp>
#include <zipper/utils/suitesparse/spqr.hpp>

using Cholmod3 = zipper::utils::suitesparse::CholmodResult<3>;
using CholmodX = zipper::utils::suitesparse::CholmodResult<>;
using Umfpack3 = zipper::utils::suitesparse::UmfpackResult<3>;
using UmfpackX = zipper::utils::suitesparse::UmfpackResult<>;
using Spqr3 = zipper::utils::suitesparse::SpqrResult<3>;
using SpqrX = zipper::utils::suitesparse::SpqrResult<>;

TEST_CASE("direct_solver_concept_suitesparse", "[concepts]") {
    STATIC_CHECK(zipper::concepts::DirectSolver<Cholmod3>);
    STATIC_CHECK(zipper::concepts::DirectSolver<CholmodX>);
    STATIC_CHECK(zipper::concepts::DirectSolver<Umfpack3>);
    STATIC_CHECK(zipper::concepts::DirectSolver<UmfpackX>);
    STATIC_CHECK(zipper::concepts::DirectSolver<Spqr3>);
    STATIC_CHECK(zipper::concepts::DirectSolver<SpqrX>);
}
#endif // ZIPPER_HAS_SUITESPARSE

// -----------------------------------------------------------------------
// Preconditioner concept
// -----------------------------------------------------------------------

TEST_CASE("preconditioner_concept", "[concepts]") {
    STATIC_CHECK(zipper::concepts::Preconditioner<JacobiP3>);
    STATIC_CHECK(zipper::concepts::Preconditioner<JacobiPX>);
    STATIC_CHECK(zipper::concepts::Preconditioner<SSORP3>);
    STATIC_CHECK(zipper::concepts::Preconditioner<SSORPX>);

    STATIC_CHECK_FALSE(zipper::concepts::Preconditioner<int>);
    STATIC_CHECK_FALSE(zipper::concepts::Preconditioner<double>);
    STATIC_CHECK_FALSE(zipper::concepts::Preconditioner<zipper::Matrix<double, 3, 3>>);
}

// -----------------------------------------------------------------------
// Algebraic scalar concepts: ring and field
// -----------------------------------------------------------------------

TEST_CASE("ring_concept", "[concepts][algebraic]") {
    // Signed and unsigned integer types are rings
    STATIC_CHECK(zipper::concepts::ring<int>);
    STATIC_CHECK(zipper::concepts::ring<unsigned>);
    STATIC_CHECK(zipper::concepts::ring<int64_t>);

    // Floating-point types are rings
    STATIC_CHECK(zipper::concepts::ring<float>);
    STATIC_CHECK(zipper::concepts::ring<double>);

    // Non-arithmetic types are not rings
    STATIC_CHECK_FALSE(zipper::concepts::ring<std::string>);
}

TEST_CASE("field_concept", "[concepts][algebraic]") {
    // Integer types are NOT fields (truncating division)
    STATIC_CHECK_FALSE(zipper::concepts::field<int>);
    STATIC_CHECK_FALSE(zipper::concepts::field<unsigned>);
    STATIC_CHECK_FALSE(zipper::concepts::field<int64_t>);

    // Floating-point types are fields
    STATIC_CHECK(zipper::concepts::field<float>);
    STATIC_CHECK(zipper::concepts::field<double>);

    // Non-arithmetic types are not fields
    STATIC_CHECK_FALSE(zipper::concepts::field<std::string>);
}

// -----------------------------------------------------------------------
// Module and vector_space concepts
// -----------------------------------------------------------------------

TEST_CASE("module_concept", "[concepts][algebraic]") {
    // Integer vectors are modules but not vector spaces
    STATIC_CHECK(zipper::concepts::module<zipper::Vector<int, 3>>);
    STATIC_CHECK_FALSE(zipper::concepts::vector_space<zipper::Vector<int, 3>>);

    // Floating-point vectors are both modules and vector spaces
    STATIC_CHECK(zipper::concepts::module<zipper::Vector<double, 3>>);
    STATIC_CHECK(zipper::concepts::vector_space<zipper::Vector<double, 3>>);

    // Matrices are modules (and vector spaces when over a field)
    STATIC_CHECK(zipper::concepts::module<zipper::Matrix<int, 3, 3>>);
    STATIC_CHECK_FALSE(zipper::concepts::vector_space<zipper::Matrix<int, 3, 3>>);
    STATIC_CHECK(zipper::concepts::module<zipper::Matrix<double, 3, 3>>);
    STATIC_CHECK(zipper::concepts::vector_space<zipper::Matrix<double, 3, 3>>);

    // Forms are modules
    STATIC_CHECK(zipper::concepts::module<zipper::Form<double, 3>>);
    STATIC_CHECK(zipper::concepts::vector_space<zipper::Form<double, 3>>);
    STATIC_CHECK(zipper::concepts::module<zipper::Form<int, 3>>);
    STATIC_CHECK_FALSE(zipper::concepts::vector_space<zipper::Form<int, 3>>);

    // cv-ref qualified types still satisfy the concepts
    STATIC_CHECK(zipper::concepts::module<const zipper::Vector<double, 3>>);
    STATIC_CHECK(zipper::concepts::module<zipper::Vector<double, 3>&>);
    STATIC_CHECK(zipper::concepts::module<const zipper::Vector<double, 3>&>);

    // Scalars are not modules (no value_type member)
    STATIC_CHECK_FALSE(zipper::concepts::module<int>);
    STATIC_CHECK_FALSE(zipper::concepts::module<double>);
}

TEST_CASE("module_concept_integer_vectors", "[concepts][algebraic]") {
    zipper::Vector<int, 3> a{1, 2, 3};
    zipper::Vector<int, 3> b{4, 5, 6};

    auto c = module_add(a, b);
    CHECK(c(0) == 5);
    CHECK(c(1) == 7);
    CHECK(c(2) == 9);
}

TEST_CASE("module_concept_double_vectors", "[concepts][algebraic]") {
    zipper::Vector<double, 3> a{1.0, 2.0, 3.0};
    zipper::Vector<double, 3> b{0.5, 1.5, 2.5};

    auto c = module_add(a, b);
    CHECK(c(0) == Catch::Approx(1.5));
    CHECK(c(1) == Catch::Approx(3.5));
    CHECK(c(2) == Catch::Approx(5.5));
}

// -----------------------------------------------------------------------
// Inner product space and normed space concepts
// -----------------------------------------------------------------------

TEST_CASE("inner_product_space_concept", "[concepts][algebraic]") {
    // Floating-point vectors have dot — satisfy inner_product_space
    STATIC_CHECK(zipper::concepts::inner_product_space<zipper::Vector<double, 3>>);

    // Matrices do NOT have dot() — not inner product spaces
    STATIC_CHECK_FALSE(zipper::concepts::inner_product_space<zipper::Matrix<double, 3, 3>>);

    // Integer vectors are not vector spaces, so not inner product spaces
    STATIC_CHECK_FALSE(zipper::concepts::inner_product_space<zipper::Vector<int, 3>>);
}

TEST_CASE("inner_product_space_concept_dot", "[concepts][algebraic]") {
    zipper::Vector<double, 3> a{1.0, 2.0, 3.0};
    zipper::Vector<double, 3> b{4.0, 5.0, 6.0};

    CHECK(ip_dot(a, b) == Catch::Approx(32.0));
}

TEST_CASE("normed_space_concept", "[concepts][algebraic]") {
    // Floating-point vectors have norm — satisfy normed_space
    STATIC_CHECK(zipper::concepts::normed_space<zipper::Vector<double, 3>>);

    // Integer vectors are not vector spaces, so not normed spaces
    STATIC_CHECK_FALSE(zipper::concepts::normed_space<zipper::Vector<int, 3>>);
}

TEST_CASE("normed_space_concept_norm", "[concepts][algebraic]") {
    zipper::Vector<double, 3> v{3.0, 4.0, 0.0};

    CHECK(ns_norm(v) == Catch::Approx(5.0));
}
