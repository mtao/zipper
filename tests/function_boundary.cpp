/// @file test_function_boundary.cpp
/// Comprehensive ergonomic tests for function boundary / expression packing
/// semantics.  These tests verify that:
///
/// - Owning types (Vector, Matrix, rvalue-STL wrappers) can be returned
/// - Borrowing types that store expressions BY VALUE (lvalue-STL wrappers,
///   expression templates like a+b) CANNOT be copied (NonReturnable) but
///   CAN be moved (to support operator chaining of temporaries)
/// - Borrowing types that store expressions BY REFERENCE (as_* wrappers)
///   ARE technically copyable but still have stores_references == true
/// - to_owned() makes borrowing types returnable
/// - unsafe() makes borrowing types returnable (caller asserts lifetime)
/// - eval() always produces a returnable owning type
/// - Packing into structs works for owning types, not for borrowing
/// - as_* results are reference-wrapping (stores_references == true)
/// - Binary/unary expressions storing references are NonReturnable
/// - Concept satisfaction for STL-deduced wrappers
/// - Mixing STL-deduced wrappers with MDArray-based types in expressions

#include "catch_include.hpp"
#include <array>
#include <type_traits>
#include <vector>

#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/as.hpp>
#include <zipper/concepts/Zipper.hpp>
#include <zipper/expression/unary/UnsafeRef.hpp>

using namespace zipper;

// ============================================================================
// Section 1: Owning types CAN be returned from functions
// ============================================================================

TEST_CASE("owning Vector can be returned from function",
          "[boundary][owning][return]") {
    auto make_vec = []() -> Vector<double, 3> {
        return Vector<double, 3>({1.0, 2.0, 3.0});
    };
    auto v = make_vec();
    REQUIRE(v(0) == 1.0);
    REQUIRE(v(1) == 2.0);
    REQUIRE(v(2) == 3.0);
}

TEST_CASE("owning Matrix can be returned from function",
          "[boundary][owning][return]") {
    auto make_mat = []() -> Matrix<double, 2, 2> {
        return Matrix<double, 2, 2>({{1.0, 2.0}, {3.0, 4.0}});
    };
    auto m = make_mat();
    REQUIRE(m(0, 0) == 1.0);
    REQUIRE(m(0, 1) == 2.0);
    REQUIRE(m(1, 0) == 3.0);
    REQUIRE(m(1, 1) == 4.0);
}

TEST_CASE("rvalue STL VectorBase is owning and can be returned",
          "[boundary][owning][return][stl]") {
    // rvalue STL -> StlMDArray<S> by value -> stores_references == false
    auto make_stl_vec = []() {
        return VectorBase(std::vector<double>{10.0, 20.0, 30.0});
    };
    auto v = make_stl_vec();
    REQUIRE(v(0) == 10.0);
    REQUIRE(v(1) == 20.0);
    REQUIRE(v(2) == 30.0);
}

TEST_CASE("rvalue STL MatrixBase is owning and can be returned",
          "[boundary][owning][return][stl]") {
    auto make_stl_mat = []() {
        return MatrixBase(
            std::vector<std::array<double, 2>>{{1.0, 2.0}, {3.0, 4.0}});
    };
    auto m = make_stl_mat();
    REQUIRE(m(0, 0) == 1.0);
    REQUIRE(m(0, 1) == 2.0);
    REQUIRE(m(1, 0) == 3.0);
    REQUIRE(m(1, 1) == 4.0);
}

// ============================================================================
// Section 2: Borrowing types CANNOT be copied/moved (NonReturnable)
// ============================================================================

TEST_CASE("lvalue STL VectorBase is NonReturnable",
          "[boundary][borrowing][static]") {
    std::vector<double> data{1.0, 2.0, 3.0};
    using BorrowingVec = decltype(VectorBase(data));

    // The wrapper stores a reference to the STL container
    STATIC_REQUIRE(BorrowingVec::stores_references);

    // Cannot copy — NonReturnable deletes copy constructor
    STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<BorrowingVec>);
    // Move is allowed (for operator chaining of temporaries)
    STATIC_REQUIRE(std::is_move_constructible_v<BorrowingVec>);
}

TEST_CASE("as_array result stores references", "[boundary][borrowing][as]") {
    Vector<double, 3> v({1.0, 2.0, 3.0});
    auto arr = as_array(v);
    using AsArrayType = decltype(arr);

    // as_array stores a reference to the expression
    STATIC_REQUIRE(AsArrayType::stores_references);

    // Note: as_* wrappers hold Expression by reference (Expr&), so the wrapper
    // itself is technically copyable (references can be copied).  The safety
    // comes from the stores_references flag being true, which tells the user
    // this is a borrowing view.  The NonReturnable mechanism applies to
    // expressions stored BY VALUE (e.g., binary expressions, lvalue STL
    // wrappers) whose expression node inherits NonReturnable.

    // Element access works through the reference
    REQUIRE(arr(0) == 1.0);
    REQUIRE(arr(1) == 2.0);
    REQUIRE(arr(2) == 3.0);
}

TEST_CASE("as_array reflects mutations of original",
          "[boundary][borrowing][as]") {
    Vector<double, 3> v({1.0, 2.0, 3.0});
    auto arr = as_array(v);

    // Mutate through the original
    v(0) = 42.0;
    REQUIRE(arr(0) == 42.0);
}

TEST_CASE("binary expression (a+b) is NonReturnable",
          "[boundary][borrowing][binary]") {
    Vector<double, 3> a({1.0, 2.0, 3.0});
    Vector<double, 3> b({4.0, 5.0, 6.0});
    auto sum = a + b;
    using SumType = decltype(sum);

    // Binary expressions store children by const reference
    STATIC_REQUIRE(SumType::stores_references);
    STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<SumType>);
    // Move is allowed (for operator chaining of temporaries)
    STATIC_REQUIRE(std::is_move_constructible_v<SumType>);

    // But lazy evaluation still works
    REQUIRE(sum(0) == 5.0);
    REQUIRE(sum(1) == 7.0);
    REQUIRE(sum(2) == 9.0);
}

TEST_CASE("scalar-expression (v * 2) is NonReturnable",
          "[boundary][borrowing][scalar]") {
    Vector<double, 3> v({1.0, 2.0, 3.0});
    auto scaled = v * 2.0;
    using ScaledType = decltype(scaled);

    STATIC_REQUIRE(ScaledType::stores_references);
    STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<ScaledType>);
    // Move is allowed (for operator chaining of temporaries)
    STATIC_REQUIRE(std::is_move_constructible_v<ScaledType>);

    REQUIRE(scaled(0) == 2.0);
    REQUIRE(scaled(1) == 4.0);
    REQUIRE(scaled(2) == 6.0);
}

TEST_CASE("unary negation is NonReturnable",
          "[boundary][borrowing][unary]") {
    Vector<double, 3> v({1.0, 2.0, 3.0});
    auto neg = -v;
    using NegType = decltype(neg);

    STATIC_REQUIRE(NegType::stores_references);
    STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<NegType>);
    // Move is allowed (for operator chaining of temporaries)
    STATIC_REQUIRE(std::is_move_constructible_v<NegType>);

    REQUIRE(neg(0) == -1.0);
    REQUIRE(neg(1) == -2.0);
    REQUIRE(neg(2) == -3.0);
}

// ============================================================================
// Section 3: Owning types have stores_references == false
// ============================================================================

TEST_CASE("concrete Vector does not store references",
          "[boundary][owning][traits]") {
    STATIC_REQUIRE_FALSE(Vector<double, 3>::stores_references);
    STATIC_REQUIRE(std::is_copy_constructible_v<Vector<double, 3>>);
    STATIC_REQUIRE(std::is_move_constructible_v<Vector<double, 3>>);
}

TEST_CASE("concrete Matrix does not store references",
          "[boundary][owning][traits]") {
    STATIC_REQUIRE_FALSE(Matrix<double, 2, 2>::stores_references);
    STATIC_REQUIRE(std::is_copy_constructible_v<Matrix<double, 2, 2>>);
    STATIC_REQUIRE(std::is_move_constructible_v<Matrix<double, 2, 2>>);
}

TEST_CASE("rvalue STL deduced VectorBase does not store references",
          "[boundary][owning][traits][stl]") {
    using OwningStlVec = decltype(VectorBase(std::vector<double>{1.0}));
    STATIC_REQUIRE_FALSE(OwningStlVec::stores_references);
    STATIC_REQUIRE(std::is_copy_constructible_v<OwningStlVec>);
    STATIC_REQUIRE(std::is_move_constructible_v<OwningStlVec>);
}

// ============================================================================
// Section 4: to_owned() makes borrowing types returnable
// ============================================================================

TEST_CASE("to_owned on lvalue STL wrapper produces returnable type",
          "[boundary][to_owned][stl]") {
    std::vector<double> data{1.0, 2.0, 3.0};
    auto borrowed = VectorBase(data);

    // borrowed is NonReturnable
    STATIC_REQUIRE(decltype(borrowed)::stores_references);

    auto owned = borrowed.to_owned();
    using OwnedType = decltype(owned);

    // owned is Returnable
    STATIC_REQUIRE_FALSE(OwnedType::stores_references);
    STATIC_REQUIRE(std::is_copy_constructible_v<OwnedType>);

    REQUIRE(owned(0) == 1.0);
    REQUIRE(owned(1) == 2.0);
    REQUIRE(owned(2) == 3.0);

    // Mutation of original does NOT affect owned copy
    data[0] = 999.0;
    REQUIRE(owned(0) == 1.0);
}

TEST_CASE("to_owned can be returned from a function",
          "[boundary][to_owned][return]") {
    Vector<double, 3> v({1.0, 2.0, 3.0});

    auto make_owned_expr = [&v]() {
        auto expr = v + v;  // NonReturnable binary expression
        return expr.to_owned();
    };

    auto result = make_owned_expr();
    STATIC_REQUIRE_FALSE(decltype(result)::stores_references);
    REQUIRE(result(0) == 2.0);
    REQUIRE(result(1) == 4.0);
    REQUIRE(result(2) == 6.0);
}

TEST_CASE("to_owned on as_array produces returnable type",
          "[boundary][to_owned][as]") {
    Vector<double, 3> v({10.0, 20.0, 30.0});
    auto arr = as_array(v);

    // arr is NonReturnable
    STATIC_REQUIRE(decltype(arr)::stores_references);

    auto owned_arr = arr.to_owned();
    STATIC_REQUIRE_FALSE(decltype(owned_arr)::stores_references);
    STATIC_REQUIRE(std::is_copy_constructible_v<decltype(owned_arr)>);

    REQUIRE(owned_arr(0) == 10.0);
    REQUIRE(owned_arr(1) == 20.0);
    REQUIRE(owned_arr(2) == 30.0);
}

// ============================================================================
// Section 5: unsafe() makes borrowing types returnable
// ============================================================================

TEST_CASE("unsafe on binary expression is returnable",
          "[boundary][unsafe][return]") {
    Vector<double, 3> a({1.0, 2.0, 3.0});
    Vector<double, 3> b({4.0, 5.0, 6.0});

    // a + b is NonReturnable
    auto sum = a + b;
    STATIC_REQUIRE(decltype(sum)::stores_references);

    auto ur = sum.unsafe();
    STATIC_REQUIRE_FALSE(decltype(ur)::stores_references);
    STATIC_REQUIRE(std::is_copy_constructible_v<decltype(ur)>);

    REQUIRE(ur(0) == 5.0);
    REQUIRE(ur(1) == 7.0);
    REQUIRE(ur(2) == 9.0);
}

TEST_CASE("unsafe can be returned from lambda",
          "[boundary][unsafe][return]") {
    Vector<double, 3> v({1.0, 2.0, 3.0});

    auto make_ref = [&v]() { return v.unsafe(); };
    auto ur = make_ref();

    REQUIRE(ur(0) == 1.0);
    REQUIRE(ur(1) == 2.0);
    REQUIRE(ur(2) == 3.0);

    // Reflects mutations (still a reference under the hood)
    v(0) = 42.0;
    REQUIRE(ur(0) == 42.0);
}

// ============================================================================
// Section 6: eval() always produces a returnable owning type
// ============================================================================

TEST_CASE("eval on binary expression produces owning Vector",
          "[boundary][eval][return]") {
    Vector<double, 3> a({1.0, 2.0, 3.0});
    Vector<double, 3> b({10.0, 20.0, 30.0});

    auto make_sum = [&a, &b]() -> Vector<double, 3> {
        auto sum = a + b;
        return sum.eval();
    };

    auto result = make_sum();
    STATIC_REQUIRE_FALSE(decltype(result)::stores_references);
    REQUIRE(result(0) == 11.0);
    REQUIRE(result(1) == 22.0);
    REQUIRE(result(2) == 33.0);

    // Mutation of originals does NOT affect eval result
    a(0) = 999.0;
    REQUIRE(result(0) == 11.0);
}

TEST_CASE("eval on STL-deduced wrapper produces owning Vector",
          "[boundary][eval][stl]") {
    std::vector<double> data{5.0, 6.0, 7.0};
    auto borrowed = VectorBase(data);

    auto evaled = borrowed.eval();
    STATIC_REQUIRE_FALSE(decltype(evaled)::stores_references);
    STATIC_REQUIRE(std::is_copy_constructible_v<decltype(evaled)>);

    REQUIRE(evaled(0) == 5.0);
    REQUIRE(evaled(1) == 6.0);
    REQUIRE(evaled(2) == 7.0);

    // Mutation of original does NOT affect eval result
    data[0] = 999.0;
    REQUIRE(evaled(0) == 5.0);
}

TEST_CASE("eval on scalar expression produces owning Vector",
          "[boundary][eval][scalar]") {
    Vector<double, 3> v({2.0, 4.0, 6.0});
    auto scaled = v * 0.5;

    // scaled is NonReturnable
    STATIC_REQUIRE(decltype(scaled)::stores_references);

    auto result = scaled.eval();
    STATIC_REQUIRE_FALSE(decltype(result)::stores_references);
    REQUIRE(result(0) == 1.0);
    REQUIRE(result(1) == 2.0);
    REQUIRE(result(2) == 3.0);
}

// ============================================================================
// Section 7: Packing into structs
// ============================================================================

namespace {
struct OwningBundle {
    Vector<double, 3> position;
    Vector<double, 3> velocity;
};
}  // namespace

TEST_CASE("owning types can be packed into structs",
          "[boundary][struct][owning]") {
    OwningBundle bundle{
        Vector<double, 3>({1.0, 2.0, 3.0}),
        Vector<double, 3>({4.0, 5.0, 6.0}),
    };

    REQUIRE(bundle.position(0) == 1.0);
    REQUIRE(bundle.velocity(2) == 6.0);

    // Struct is copyable
    auto bundle2 = bundle;
    REQUIRE(bundle2.position(0) == 1.0);

    // Independent copy
    bundle.position(0) = 999.0;
    REQUIRE(bundle2.position(0) == 1.0);
}

TEST_CASE("borrowing types cannot be struct members (static verification)",
          "[boundary][struct][borrowing]") {
    // We can't actually create a struct with NonReturnable members because the
    // struct's copy/move ctors would be deleted.  Verify the fundamental
    // property that makes this impossible.
    std::vector<double> data{1.0, 2.0};
    using BorrowingVec = decltype(VectorBase(data));

    // A struct containing this type would inherit non-copyability
    STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<BorrowingVec>);
    // Move is allowed (for operator chaining of temporaries)
    STATIC_REQUIRE(std::is_move_constructible_v<BorrowingVec>);
}

// ============================================================================
// Section 8: Concept satisfaction for STL-deduced wrappers
// ============================================================================

TEST_CASE("rvalue STL VectorBase satisfies Vector concept",
          "[boundary][concepts][stl]") {
    using RvalueStlVec = decltype(VectorBase(std::vector<double>{1.0}));
    STATIC_REQUIRE(concepts::Vector<RvalueStlVec>);
    STATIC_REQUIRE(concepts::Zipper<RvalueStlVec>);
}

TEST_CASE("lvalue STL VectorBase satisfies Vector concept",
          "[boundary][concepts][stl]") {
    std::vector<double> data{1.0};
    using LvalueStlVec = decltype(VectorBase(data));
    STATIC_REQUIRE(concepts::Vector<LvalueStlVec>);
    STATIC_REQUIRE(concepts::Zipper<LvalueStlVec>);
}

TEST_CASE("rvalue STL MatrixBase satisfies Matrix concept",
          "[boundary][concepts][stl]") {
    using RvalueStlMat =
        decltype(MatrixBase(std::vector<std::array<double, 2>>{{1.0, 2.0}}));
    STATIC_REQUIRE(concepts::Matrix<RvalueStlMat>);
    STATIC_REQUIRE(concepts::Zipper<RvalueStlMat>);
}

TEST_CASE("lvalue STL MatrixBase satisfies Matrix concept",
          "[boundary][concepts][stl]") {
    std::vector<std::array<double, 2>> data{{1.0, 2.0}};
    using LvalueStlMat = decltype(MatrixBase(data));
    STATIC_REQUIRE(concepts::Matrix<LvalueStlMat>);
    STATIC_REQUIRE(concepts::Zipper<LvalueStlMat>);
}

// ============================================================================
// Section 9: Mixing STL-deduced wrappers with MDArray types in expressions
// ============================================================================

TEST_CASE("STL VectorBase + concrete Vector arithmetic",
          "[boundary][mixing][stl]") {
    std::array<double, 3> arr = {1.0, 2.0, 3.0};
    auto stl_vec = VectorBase(arr);
    Vector<double, 3> concrete({10.0, 20.0, 30.0});

    auto sum = stl_vec + concrete;
    REQUIRE(sum(0) == 11.0);
    REQUIRE(sum(1) == 22.0);
    REQUIRE(sum(2) == 33.0);

    // The expression is NonReturnable (stores references)
    STATIC_REQUIRE(decltype(sum)::stores_references);

    // But eval() materializes it
    auto result = sum.eval();
    STATIC_REQUIRE_FALSE(decltype(result)::stores_references);
    REQUIRE(result(0) == 11.0);
}

TEST_CASE("rvalue STL VectorBase + concrete Vector arithmetic",
          "[boundary][mixing][stl]") {
    auto stl_vec = VectorBase(std::array<double, 3>{1.0, 2.0, 3.0});
    Vector<double, 3> concrete({10.0, 20.0, 30.0});

    auto sum = stl_vec + concrete;
    REQUIRE(sum(0) == 11.0);
    REQUIRE(sum(1) == 22.0);
    REQUIRE(sum(2) == 33.0);
}

TEST_CASE("scalar multiply on STL VectorBase",
          "[boundary][mixing][stl][scalar]") {
    std::array<double, 3> arr = {2.0, 4.0, 6.0};
    auto stl_vec = VectorBase(arr);

    auto scaled = stl_vec * 0.5;
    REQUIRE(scaled(0) == 1.0);
    REQUIRE(scaled(1) == 2.0);
    REQUIRE(scaled(2) == 3.0);
}

// ============================================================================
// Section 10: Chained operations and complex scenarios
// ============================================================================

TEST_CASE("chained expression eval in single statement",
          "[boundary][chain][eval]") {
    Vector<double, 3> a({1.0, 2.0, 3.0});
    Vector<double, 3> b({4.0, 5.0, 6.0});

    // Chaining must be evaluated in a single statement to avoid dangling
    // references to temporaries.  This is the standard expression template
    // pattern: build and consume in one go.
    auto result = ((a + b) * 2.0).eval();
    STATIC_REQUIRE_FALSE(decltype(result)::stores_references);
    REQUIRE(result(0) == 10.0);
    REQUIRE(result(1) == 14.0);
    REQUIRE(result(2) == 18.0);
}

TEST_CASE("eval result can be used in further expressions",
          "[boundary][chain][eval]") {
    Vector<double, 3> a({1.0, 2.0, 3.0});
    Vector<double, 3> b({4.0, 5.0, 6.0});

    auto sum_owned = (a + b).eval();
    // sum_owned is owning, can participate in new expressions
    auto doubled = sum_owned + sum_owned;
    REQUIRE(doubled(0) == 10.0);
    REQUIRE(doubled(1) == 14.0);
    REQUIRE(doubled(2) == 18.0);
}

TEST_CASE("to_owned preserves lazy evaluation semantics",
          "[boundary][to_owned][lazy]") {
    Vector<double, 3> a({1.0, 2.0, 3.0});
    Vector<double, 3> b({10.0, 20.0, 30.0});

    auto sum = a + b;
    auto owned = sum.to_owned();

    // owned should have the computed values baked in (children are now owned)
    REQUIRE(owned(0) == 11.0);
    REQUIRE(owned(1) == 22.0);
    REQUIRE(owned(2) == 33.0);

    // Original mutation does NOT affect owned
    a(0) = 999.0;
    REQUIRE(owned(0) == 11.0);
}

TEST_CASE("function returning eval of complex expression",
          "[boundary][function][eval]") {
    auto compute = [](const Vector<double, 3> &x,
                      const Vector<double, 3> &y) -> Vector<double, 3> {
        return ((x + y) * 2.0).eval();
    };

    Vector<double, 3> x({1.0, 2.0, 3.0});
    Vector<double, 3> y({4.0, 5.0, 6.0});
    auto result = compute(x, y);

    REQUIRE(result(0) == 10.0);
    REQUIRE(result(1) == 14.0);
    REQUIRE(result(2) == 18.0);
}

// ============================================================================
// Section 11: Rvalue-safe operator chaining (Phase 7)
// ============================================================================

TEST_CASE("chained (a+b)*2 stored in variable, no dangling",
          "[boundary][chain][rvalue]") {
    Vector<double, 3> a({1.0, 2.0, 3.0});
    Vector<double, 3> b({4.0, 5.0, 6.0});

    // Previously this would segfault because (a+b) creates a temporary wrapper,
    // and * 2.0 would store a dangling reference to it.  With Phase 7 rvalue
    // chaining, the intermediate (a+b) expression is MOVED into the * node.
    auto chain = (a + b) * 2.0;

    // The chain still stores references to the leaf lvalues a and b
    STATIC_REQUIRE(decltype(chain)::stores_references);

    // But evaluating later (after the temporary is gone) should work
    REQUIRE(chain(0) == 10.0);
    REQUIRE(chain(1) == 14.0);
    REQUIRE(chain(2) == 18.0);
}

TEST_CASE("chained 2*(a+b) stored in variable, scalar on left",
          "[boundary][chain][rvalue]") {
    Vector<double, 3> a({1.0, 2.0, 3.0});
    Vector<double, 3> b({4.0, 5.0, 6.0});

    auto chain = 2.0 * (a + b);

    STATIC_REQUIRE(decltype(chain)::stores_references);
    REQUIRE(chain(0) == 10.0);
    REQUIRE(chain(1) == 14.0);
    REQUIRE(chain(2) == 18.0);
}

TEST_CASE("chained -(a+b) stored in variable, unary on rvalue",
          "[boundary][chain][rvalue]") {
    Vector<double, 3> a({1.0, 2.0, 3.0});
    Vector<double, 3> b({4.0, 5.0, 6.0});

    auto chain = -(a + b);

    STATIC_REQUIRE(decltype(chain)::stores_references);
    REQUIRE(chain(0) == -5.0);
    REQUIRE(chain(1) == -7.0);
    REQUIRE(chain(2) == -9.0);
}

TEST_CASE("triple chain (a+b)*2+c stored in variable",
          "[boundary][chain][rvalue]") {
    Vector<double, 3> a({1.0, 2.0, 3.0});
    Vector<double, 3> b({4.0, 5.0, 6.0});
    Vector<double, 3> c({100.0, 200.0, 300.0});

    auto chain = (a + b) * 2.0 + c;

    STATIC_REQUIRE(decltype(chain)::stores_references);
    REQUIRE(chain(0) == 110.0);
    REQUIRE(chain(1) == 214.0);
    REQUIRE(chain(2) == 318.0);
}

TEST_CASE("chained expression eval produces correct owned result",
          "[boundary][chain][rvalue][eval]") {
    Vector<double, 3> a({1.0, 2.0, 3.0});
    Vector<double, 3> b({4.0, 5.0, 6.0});

    auto chain = (a + b) * 2.0;
    auto result = chain.eval();

    STATIC_REQUIRE_FALSE(decltype(result)::stores_references);
    REQUIRE(result(0) == 10.0);
    REQUIRE(result(1) == 14.0);
    REQUIRE(result(2) == 18.0);

    // Original mutation does NOT affect eval result
    a(0) = 999.0;
    REQUIRE(result(0) == 10.0);
}

TEST_CASE("chained expression reflects leaf mutation (lazy)",
          "[boundary][chain][rvalue][lazy]") {
    Vector<double, 3> a({1.0, 2.0, 3.0});
    Vector<double, 3> b({4.0, 5.0, 6.0});

    auto chain = (a + b) * 2.0;
    REQUIRE(chain(0) == 10.0);

    // Since chain still stores references to a and b, mutation is visible
    a(0) = 10.0;
    REQUIRE(chain(0) == 28.0);  // (10+4)*2 = 28
}

TEST_CASE("deeply chained (-(a+b)*2+c)*0.5 stored in variable",
          "[boundary][chain][rvalue][deep]") {
    Vector<double, 3> a({1.0, 2.0, 3.0});
    Vector<double, 3> b({4.0, 5.0, 6.0});
    Vector<double, 3> c({100.0, 200.0, 300.0});

    auto chain = (-(a + b) * 2.0 + c) * 0.5;

    STATIC_REQUIRE(decltype(chain)::stores_references);
    // -(a+b)*2 = {-10, -14, -18}, +c = {90, 186, 282}, *0.5 = {45, 93, 141}
    REQUIRE(chain(0) == 45.0);
    REQUIRE(chain(1) == 93.0);
    REQUIRE(chain(2) == 141.0);
}

TEST_CASE("rvalue+rvalue: (a+b)+(c+d) chained",
          "[boundary][chain][rvalue]") {
    Vector<double, 3> a({1.0, 2.0, 3.0});
    Vector<double, 3> b({4.0, 5.0, 6.0});
    Vector<double, 3> c({10.0, 20.0, 30.0});
    Vector<double, 3> d({100.0, 200.0, 300.0});

    auto chain = (a + b) + (c + d);

    STATIC_REQUIRE(decltype(chain)::stores_references);
    REQUIRE(chain(0) == 115.0);
    REQUIRE(chain(1) == 227.0);
    REQUIRE(chain(2) == 339.0);
}

// ============================================================================
// Section 12: Implicit squeeze / degenerate-dimension slicing in as_*
// ============================================================================

TEST_CASE("as_vector on Matrix<double,3,1> squeezes column dimension",
          "[squeeze][as_vector][matrix]") {
    Matrix<double, 3, 1> m({{1.0}, {2.0}, {3.0}});

    auto v = as_vector(m);

    // Result should be a rank-1 VectorBase view
    STATIC_REQUIRE(decltype(v)::expression_traits::extents_type::rank() == 1);
    REQUIRE(v.extent(0) == 3);
    REQUIRE(v(0) == 1.0);
    REQUIRE(v(1) == 2.0);
    REQUIRE(v(2) == 3.0);
}

TEST_CASE("as_vector on Matrix<double,1,3> squeezes row dimension",
          "[squeeze][as_vector][matrix]") {
    Matrix<double, 1, 3> m({{1.0, 2.0, 3.0}});

    auto v = as_vector(m);

    STATIC_REQUIRE(decltype(v)::expression_traits::extents_type::rank() == 1);
    REQUIRE(v.extent(0) == 3);
    REQUIRE(v(0) == 1.0);
    REQUIRE(v(1) == 2.0);
    REQUIRE(v(2) == 3.0);
}

TEST_CASE("as_vector on already-rank-1 Vector passes through",
          "[squeeze][as_vector][identity]") {
    Vector<double, 3> orig({10.0, 20.0, 30.0});

    auto v = as_vector(orig);

    STATIC_REQUIRE(decltype(v)::expression_traits::extents_type::rank() == 1);
    REQUIRE(v(0) == 10.0);
    REQUIRE(v(1) == 20.0);
    REQUIRE(v(2) == 30.0);
}

TEST_CASE("as_vector squeeze preserves mutability on non-const source",
          "[squeeze][as_vector][mutable]") {
    Matrix<double, 3, 1> m({{1.0}, {2.0}, {3.0}});

    auto v = as_vector(m);

    // Mutate through the squeezed view
    v(0) = 42.0;
    REQUIRE(m(0, 0) == 42.0);

    // Mutate original and see it through the view
    m(1, 0) = 99.0;
    REQUIRE(v(1) == 99.0);
}

TEST_CASE("as_vector squeeze on const source is read-only",
          "[squeeze][as_vector][const]") {
    const Matrix<double, 3, 1> m({{1.0}, {2.0}, {3.0}});

    auto v = as_vector(m);

    REQUIRE(v(0) == 1.0);
    REQUIRE(v(1) == 2.0);
    REQUIRE(v(2) == 3.0);
}

TEST_CASE("as_matrix on Tensor<double,3,1,4> squeezes degenerate dim",
          "[squeeze][as_matrix][tensor]") {
    // Rank-3 tensor with middle dimension degenerate (extent 1).
    // squeeze_rank == 2 (dims 0 and 2 are non-degenerate).
    Tensor<double, 3, 1, 4> t;

    // Fill with known values: t(i, 0, k) = i * 10 + k
    for (index_type i = 0; i < 3; ++i) {
        for (index_type k = 0; k < 4; ++k) {
            t(i, 0, k) = static_cast<double>(i * 10 + k);
        }
    }

    auto mat = as_matrix(t);

    STATIC_REQUIRE(decltype(mat)::expression_traits::extents_type::rank() == 2);
    REQUIRE(mat.extent(0) == 3);
    REQUIRE(mat.extent(1) == 4);

    for (index_type i = 0; i < 3; ++i) {
        for (index_type k = 0; k < 4; ++k) {
            REQUIRE(mat(i, k) == static_cast<double>(i * 10 + k));
        }
    }
}

TEST_CASE("as_matrix on already-rank-2 Matrix passes through",
          "[squeeze][as_matrix][identity]") {
    Matrix<double, 2, 3> m({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});

    auto mat = as_matrix(m);

    STATIC_REQUIRE(decltype(mat)::expression_traits::extents_type::rank() == 2);
    REQUIRE(mat(0, 0) == 1.0);
    REQUIRE(mat(0, 2) == 3.0);
    REQUIRE(mat(1, 1) == 5.0);
}

TEST_CASE("as_matrix squeeze preserves mutability",
          "[squeeze][as_matrix][mutable]") {
    Tensor<double, 3, 1, 4> t;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type k = 0; k < 4; ++k) {
            t(i, 0, k) = 0.0;
        }
    }

    auto mat = as_matrix(t);

    // Mutate through the squeezed view
    mat(1, 2) = 42.0;
    REQUIRE(t(1, 0, 2) == 42.0);

    // Mutate original and see it through the view
    t(0, 0, 3) = 99.0;
    REQUIRE(mat(0, 3) == 99.0);
}

TEST_CASE("as_vector on Tensor<double,1,5,1> squeezes two degenerate dims",
          "[squeeze][as_vector][tensor]") {
    // Rank-3 tensor where dims 0 and 2 are degenerate.
    // squeeze_rank == 1 (only dim 1 is non-degenerate).
    Tensor<double, 1, 5, 1> t;

    for (index_type j = 0; j < 5; ++j) {
        t(0, j, 0) = static_cast<double>(j * j);
    }

    auto v = as_vector(t);

    STATIC_REQUIRE(decltype(v)::expression_traits::extents_type::rank() == 1);
    REQUIRE(v.extent(0) == 5);
    REQUIRE(v(0) == 0.0);
    REQUIRE(v(1) == 1.0);
    REQUIRE(v(2) == 4.0);
    REQUIRE(v(3) == 9.0);
    REQUIRE(v(4) == 16.0);
}

TEST_CASE("as_vector squeeze result stores references",
          "[squeeze][as_vector][references]") {
    Matrix<double, 3, 1> m({{1.0}, {2.0}, {3.0}});

    auto v = as_vector(m);

    // The squeezed view is a borrowing wrapper (stores references)
    STATIC_REQUIRE(decltype(v)::stores_references);
}

TEST_CASE("as_vector squeeze result can be eval'd to owning Vector",
          "[squeeze][as_vector][eval]") {
    Matrix<double, 3, 1> m({{1.0}, {2.0}, {3.0}});

    auto v = as_vector(m);
    auto owned = v.eval();

    STATIC_REQUIRE_FALSE(decltype(owned)::stores_references);
    REQUIRE(owned(0) == 1.0);
    REQUIRE(owned(1) == 2.0);
    REQUIRE(owned(2) == 3.0);

    // Mutation of original does not affect eval'd copy
    m(0, 0) = 999.0;
    REQUIRE(owned(0) == 1.0);
}
