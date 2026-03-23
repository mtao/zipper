#include <zipper/Array.hpp>
#include <zipper/Form.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/concepts/DirectSolver.hpp>
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
// Static assertions: concrete types satisfy their specific concepts
// AND the umbrella Zipper concept.
// -----------------------------------------------------------------------

// Vector
static_assert(zipper::concepts::Vector<zipper::Vector<double, 3>>);
static_assert(zipper::concepts::Zipper<zipper::Vector<double, 3>>);
static_assert(zipper::concepts::Zipper<const zipper::Vector<double, 3>>);
static_assert(zipper::concepts::Zipper<zipper::Vector<double, 3>&>);
static_assert(zipper::concepts::Zipper<const zipper::Vector<double, 3>&>);

// Matrix
static_assert(zipper::concepts::Matrix<zipper::Matrix<double, 3, 3>>);
static_assert(zipper::concepts::Zipper<zipper::Matrix<double, 3, 3>>);
static_assert(zipper::concepts::Zipper<const zipper::Matrix<double, 3, 3>>);
static_assert(zipper::concepts::Zipper<zipper::Matrix<double, 3, 3>&>);
static_assert(zipper::concepts::Zipper<const zipper::Matrix<double, 3, 3>&>);

// Tensor
using Tensor3 = zipper::Tensor_<double, zipper::extents<2, 3, 4>>;
static_assert(zipper::concepts::Tensor<Tensor3>);
static_assert(zipper::concepts::Zipper<Tensor3>);
static_assert(zipper::concepts::Zipper<const Tensor3>);
static_assert(zipper::concepts::Zipper<Tensor3&>);
static_assert(zipper::concepts::Zipper<const Tensor3&>);

// Array
using Array2 = zipper::Array<double, 3, 4>;
static_assert(zipper::concepts::Array<Array2>);
static_assert(zipper::concepts::Zipper<Array2>);
static_assert(zipper::concepts::Zipper<const Array2>);
static_assert(zipper::concepts::Zipper<Array2&>);
static_assert(zipper::concepts::Zipper<const Array2&>);

// Form
using Form1 = zipper::Form<double, 3>;
static_assert(zipper::concepts::Form<Form1>);
static_assert(zipper::concepts::Zipper<Form1>);
static_assert(zipper::concepts::Zipper<const Form1>);
static_assert(zipper::concepts::Zipper<Form1&>);
static_assert(zipper::concepts::Zipper<const Form1&>);

// -----------------------------------------------------------------------
// Base types also satisfy Zipper (they have Is* specializations)
// -----------------------------------------------------------------------
using VBase = zipper::VectorBase<
    zipper::expression::nullary::MDArray<double, zipper::extents<3>>>;
static_assert(zipper::concepts::Vector<VBase>);
static_assert(zipper::concepts::Zipper<VBase>);

using MBase = zipper::MatrixBase<
    zipper::expression::nullary::MDArray<double, zipper::extents<3, 3>>>;
static_assert(zipper::concepts::Matrix<MBase>);
static_assert(zipper::concepts::Zipper<MBase>);

using TBase = zipper::TensorBase<
    zipper::expression::nullary::MDArray<double, zipper::extents<2, 3, 4>>>;
static_assert(zipper::concepts::Tensor<TBase>);
static_assert(zipper::concepts::Zipper<TBase>);

using ABase = zipper::ArrayBase<
    zipper::expression::nullary::MDArray<double, zipper::extents<3, 4>>>;
static_assert(zipper::concepts::Array<ABase>);
static_assert(zipper::concepts::Zipper<ABase>);

using FBase = zipper::FormBase<
    zipper::expression::nullary::MDArray<double, zipper::extents<3>>>;
static_assert(zipper::concepts::Form<FBase>);
static_assert(zipper::concepts::Zipper<FBase>);

// -----------------------------------------------------------------------
// Negative tests: raw expressions and scalars are NOT Zipper
// -----------------------------------------------------------------------
static_assert(!zipper::concepts::Zipper<int>);
static_assert(!zipper::concepts::Zipper<double>);
static_assert(!zipper::concepts::Zipper<
              zipper::expression::nullary::MDArray<double, zipper::extents<3>>>);

// -----------------------------------------------------------------------
// Concepts are mutually exclusive where expected
// -----------------------------------------------------------------------
static_assert(!zipper::concepts::Matrix<zipper::Vector<double, 3>>);
static_assert(!zipper::concepts::Vector<zipper::Matrix<double, 3, 3>>);

// -----------------------------------------------------------------------
// DirectSolver concept: decomposition result types satisfy it
// -----------------------------------------------------------------------
using LLT3 = zipper::utils::decomposition::LLTResult<double, 3>;
using LDLT3 = zipper::utils::decomposition::LDLTResult<double, 3>;
using QRReduced33 = zipper::utils::decomposition::QRReducedResult<double, 3, 3>;
using QRFull33 = zipper::utils::decomposition::QRFullResult<double, 3, 3>;

static_assert(zipper::concepts::DirectSolver<LLT3>);
static_assert(zipper::concepts::DirectSolver<LDLT3>);
static_assert(zipper::concepts::DirectSolver<QRReduced33>);
static_assert(zipper::concepts::DirectSolver<QRFull33>);

// Negative: plain types do not satisfy DirectSolver.
static_assert(!zipper::concepts::DirectSolver<int>);
static_assert(!zipper::concepts::DirectSolver<double>);
static_assert(!zipper::concepts::DirectSolver<zipper::Matrix<double, 3, 3>>);

// TriangularView satisfies DirectSolver (has value_type + .solve(b)).
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

static_assert(zipper::concepts::DirectSolver<LowerTriView>);
static_assert(zipper::concepts::DirectSolver<UpperTriView>);
static_assert(zipper::concepts::DirectSolver<UnitLowerTriView>);

// PLUResult satisfies DirectSolver.
using PLU3 = zipper::utils::decomposition::PLUResult<double, 3>;
using PLUX = zipper::utils::decomposition::PLUResult<double, zipper::dynamic_extent>;
static_assert(zipper::concepts::DirectSolver<PLU3>);
static_assert(zipper::concepts::DirectSolver<PLUX>);

// -----------------------------------------------------------------------
// SuiteSparse result types satisfy DirectSolver (when available)
// -----------------------------------------------------------------------
#ifdef ZIPPER_HAS_SUITESPARSE
#include <zipper/utils/suitesparse/cholmod.hpp>
#include <zipper/utils/suitesparse/umfpack.hpp>
#include <zipper/utils/suitesparse/spqr.hpp>

using Cholmod3 = zipper::utils::suitesparse::CholmodResult<3>;
using CholmodX = zipper::utils::suitesparse::CholmodResult<>;
static_assert(zipper::concepts::DirectSolver<Cholmod3>);
static_assert(zipper::concepts::DirectSolver<CholmodX>);

using Umfpack3 = zipper::utils::suitesparse::UmfpackResult<3>;
using UmfpackX = zipper::utils::suitesparse::UmfpackResult<>;
static_assert(zipper::concepts::DirectSolver<Umfpack3>);
static_assert(zipper::concepts::DirectSolver<UmfpackX>);

using Spqr3 = zipper::utils::suitesparse::SpqrResult<3>;
using SpqrX = zipper::utils::suitesparse::SpqrResult<>;
static_assert(zipper::concepts::DirectSolver<Spqr3>);
static_assert(zipper::concepts::DirectSolver<SpqrX>);
#endif // ZIPPER_HAS_SUITESPARSE

// -----------------------------------------------------------------------
// Preconditioner concept: preconditioner types satisfy it
// -----------------------------------------------------------------------
using JacobiP3 = zipper::utils::solver::JacobiPreconditioner<double, 3>;
using JacobiPX = zipper::utils::solver::JacobiPreconditioner<double, zipper::dynamic_extent>;
using SSORP3 = zipper::utils::solver::SSORPreconditioner<double, 3>;
using SSORPX = zipper::utils::solver::SSORPreconditioner<double, zipper::dynamic_extent>;

static_assert(zipper::concepts::Preconditioner<JacobiP3>);
static_assert(zipper::concepts::Preconditioner<JacobiPX>);
static_assert(zipper::concepts::Preconditioner<SSORP3>);
static_assert(zipper::concepts::Preconditioner<SSORPX>);

// Negative: plain types do not satisfy Preconditioner.
static_assert(!zipper::concepts::Preconditioner<int>);
static_assert(!zipper::concepts::Preconditioner<double>);
static_assert(!zipper::concepts::Preconditioner<zipper::Matrix<double, 3, 3>>);

// -----------------------------------------------------------------------
// Runtime tests (Catch2) — instantiate objects and confirm concepts work
// at template-constrained call sites.
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

} // namespace

TEST_CASE("Zipper concept on Vector", "[concepts]") {
    zipper::Vector<double, 3> v;
    v(0) = 1.0;
    v(1) = 2.0;
    v(2) = 3.0;

    REQUIRE(zipper_rank(v) == 1);
    REQUIRE(vector_size(v) == 3);
}

TEST_CASE("Zipper concept on Matrix", "[concepts]") {
    zipper::Matrix<double, 2, 3> m;
    for (zipper::index_type r = 0; r < 2; ++r)
        for (zipper::index_type c = 0; c < 3; ++c)
            m(r, c) = static_cast<double>(r * 3 + c);

    REQUIRE(zipper_rank(m) == 2);
    REQUIRE(matrix_rows(m) == 2);
}

TEST_CASE("Zipper concept on Tensor", "[concepts]") {
    Tensor3 t;
    REQUIRE(zipper_rank(t) == 3);
}

TEST_CASE("Zipper concept on Array", "[concepts]") {
    Array2 a;
    REQUIRE(zipper_rank(a) == 2);
}

TEST_CASE("Zipper concept on Form", "[concepts]") {
    Form1 f;
    REQUIRE(zipper_rank(f) == 1);
}
