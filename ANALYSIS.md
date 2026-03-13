# Zipper Codebase Analysis

**Date:** March 12, 2026
**Branch:** `mtao/refactor_all`
**Build system:** Meson + Catch2
**Language standard:** C++26 (GCC)

---

## 1. Architecture Overview

Zipper is a C++26 tensor processing library built around expression templates with CRTP.

### Layer Model

```
┌─────────────────────────────────────────────────────┐
│  User-facing wrappers                               │
│  Matrix<T,R,C>  Vector<T,N>  Form<T,N>  Tensor<T>  │
│  (MatrixBase, VectorBase, FormBase, TensorBase)      │
├─────────────────────────────────────────────────────┤
│  ZipperBase<Derived>                                 │
│  Common interface: assign(), operator=, dot(), etc.  │
├─────────────────────────────────────────────────────┤
│  ArrayBase<Derived>                                  │
│  Dense storage integration                           │
├─────────────────────────────────────────────────────┤
│  Expression layer                                    │
│  ExpressionBase<Derived> (CRTP root)                 │
│  ├── nullary/  (Identity, Constant, MDArray, etc.)   │
│  ├── unary/    (Slice, Transpose, AntiSlice, etc.)   │
│  ├── binary/   (CWiseBinary, etc.)                   │
│  └── reduction/(Sum, Norm, Any, CoefficientProduct)  │
├─────────────────────────────────────────────────────┤
│  Storage layer                                       │
│  StlStorage, SparseCoordinate, layout policies       │
├─────────────────────────────────────────────────────┤
│  Concepts + Traits                                   │
│  concepts::Matrix, concepts::Vector, etc.            │
│  ExpressionTraits<T>, AccessFeatures, ShapeFeatures  │
└─────────────────────────────────────────────────────┘
```

### Key Patterns

- **CRTP everywhere**: `ExpressionBase<Derived>` is the root. Unary expressions inherit `UnaryExpressionBase<Derived, ChildExpr>`. Binary expressions use `BinaryExpressionBase<Derived, LHS, RHS>`.
- **Traits-based dispatch**: `ExpressionTraits<T>` specializations define `value_type`, `extents_type`, `AccessFeatures`, `ShapeFeatures`. **Important:** `ExpressionTraits<T&>` does NOT exist — must `remove_reference_t` before lookup.
- **Concept hierarchy**: `concepts::Matrix`, `concepts::Vector`, `concepts::Form`, `concepts::Zipper` are the top-level concepts checked via `Is{Matrix,Vector,Form,Zipper}` type traits. `concepts::Expression` and `RankedExpression<T,N>` only match raw expression types (like `MDArray`), **not** the wrapper types.
- **`QualifiedExpression` concept**: Handles ref-qualified types in expression templates.
- **`std::expected` for errors**: Used in the power method solver; not yet widely adopted elsewhere.

### C++26 Usage

| Feature | Usage | Count |
|---------|-------|-------|
| Pack indexing (`args...[K]`) | With `__cpp_pack_indexing` feature test | Several |
| `std::expected` | C++23, used in power_method | 1 site |
| `consteval` functions | Throughout | 76+ |
| Concepts | Core architecture | 43+ |
| Deducing this | All 6 CRTP `derived()` methods | 6 sites |
| `static operator()` | Not used | 0 |
| `std::print` | Not used | 0 |

---

## 2. Bugs and Correctness Issues

### 2.1 Dangling References in Expression Templates (HIGH) -- FIXED

**Location:** `include/zipper/expression/binary/BinaryExpressionBase.hpp`

All three expression families (binary, unary, reductions) now use `expression_storage_t<T>` from `detail/ExpressionStorage.hpp`. When child type parameters are `const T&` (lvalue ref), storage is by reference; when they are `T` (non-ref), storage is by value. All derived classes pass `const A&` / `const B&` / `const ExprType&` as child types, hinting the storage to use references for named expressions and values for temporaries.

- **BinaryExpressionBase**: refactored with forwarding constructor and `expression_storage_t`
- **UnaryExpressionBase**: refactored with forwarding constructor and `expression_storage_t`
- **ReductionBase**: new CRTP base class created (`ReductionBase.hpp`) with `expression_storage_t`
- All 9 reduction classes, 8 binary classes, and ~15 unary classes updated

### 2.2 MDArray Initialization Trap (HIGH) -- FIXED

Expressions that inherit from `extents<...>` (e.g., `Identity<double, 3, 4>`) could trigger the wrong `MDArray` constructor when the `ZipperBase` variadic fallback constructor was selected. The `MDArray(const extents_type&)` constructor is already `explicit`, but the `ZipperBase(Args&&...args)` variadic forwarding constructor could forward an expression to MDArray, where it would slice to its `extents<...>` base class and zero-initialize.

**Fix:** Added a constraint to the `ZipperBase` variadic fallback constructor (`ZipperBase.hpp:107-111`) to reject single-argument `Expression` types when `Expression` is a non-reference type. This forces expressions to go through the proper `ZipperBase(const Other&)` constructor that calls `assign()`. When `Expression` is a reference type (used by wrapper views like `as_array()`), the variadic fallback is still allowed since it just binds the reference.

### 2.3 `assert(false)` as Placeholder (MEDIUM) -- FIXED

**Location:** `include/zipper/FormBase.hpp:104`, `include/zipper/expression/unary/Slice.hpp:253`

Replaced all `assert(false)` placeholders with `std::unreachable()`, which has well-defined behavior as a hint to the optimizer and does not silently fall through in release builds.

### 2.4 Broken Return Statement (LOW — orphaned file)

**Location:** `include/zipper/storage/detail/SparseLayoutPolicy.hpp:22`

Contains `return` with no expression in a non-void function.

This file is orphaned (not included anywhere), so it doesn't cause build failures.

### 2.5 UB in `namespace std` (MEDIUM) -- FIXED

**Location:** `include/zipper/storage/SparseCoordinateAccessor.hpp`, `include/zipper/utils/extents/extents_formatter.hpp`

- `SparseCoordinateAccessor.hpp`: Removed redundant `std::distance`/`std::advance` overloads. The iterator already defines `operator-` and `operator+=`, so the standard implementations work correctly for random access iterators.
- `extents_formatter.hpp`: Replaced illegal `format_as` function in `namespace std` with a proper `fmt::formatter<zipper::extents<...>>` specialization. Removed dead `#if defined(ASDF)` code block.

### 2.6 ODR Violations (LOW) -- FIXED

All 30 `assert()` calls across 16 header files were replaced with `ZIPPER_ASSERT()`, a custom macro defined in `include/zipper/detail/assert.hpp`. Unlike `assert()`, which depends on the `NDEBUG` preprocessor symbol and can produce different code in different translation units, `ZIPPER_ASSERT()` is controlled by a zipper-specific `ZIPPER_DEBUG` macro that is set uniformly via build-system flags (e.g. `-DZIPPER_DEBUG`). When `ZIPPER_DEBUG` is not defined, the macro is a no-op with zero overhead; when defined, it evaluates the expression and aborts with a diagnostic message on failure. This eliminates ODR violations regardless of per-TU `NDEBUG` settings.

---

## 3. Code Quality Issues

### 3.1 Missing Include Guard -- FIXED

**Location:** `include/zipper/expression/nullary/StlMDArray.hpp`

Added `#pragma once`.

### 3.2 Typo in Include Guard -- FIXED

**Location:** `include/zipper/detail/Features.hpp`

Fixed `ZIPPER_DETAIL_FEATUERS_HPP` to `ZIPPER_DETAIL_FEATURES_HPP`.

### 3.3 Unnecessary Dependency -- FIXED

**Location:** `include/zipper/expression/nullary/NullaryExpressionBase.hpp:6`

Removed unused `#include <spdlog/spdlog.h>`.

### 3.4 Pragma Proliferation

129+ diagnostic suppression pragmas across the codebase, mainly for external headers. Consider isolating external includes behind wrapper headers with targeted suppressions.

---

## 4. Orphaned / Stub Files

These files are not included by any other file and contain broken or incomplete code. They were kept as stubs per project policy (except `inverse.hpp` which was deleted).

| File | Status | Notes |
|------|--------|-------|
| `expression/binary/Addition.hpp` | Orphaned | Old API, never included |
| `expression/binary/CoeffProduct.hpp` | Orphaned | Old API, never included |
| `expression/detail/ExpressionTypeHelper.hpp` | Orphaned | Unused helper |
| `expression/DynamicSizedExpressionBase.hpp` | Orphaned | Unused base class |
| `expression/binary/Contraction.hpp` | Orphaned | Never included |
| `expression/unary/DiagonalTensor.hpp` | Orphaned | Never included |
| `storage/detail/SparseLayoutPolicy.hpp` | Orphaned | Broken `return` statement |
| `utils/solvers/pivot_utils.hpp` | Orphaned | Unused utilities |

---

## 5. Naming and Organization Inconsistencies

### 5.1 Mixed Naming Conventions

- Expression headers use PascalCase (`Slice.hpp`, `Transpose.hpp`, `AntiSlice.hpp`)
- Utility headers use snake_case (`is_cubic.hpp`, `power_method.hpp`)
- Some detail headers are inconsistent

### 5.2 Old `views::` Namespace Fully Removed

The migration from `zipper::views::` to `zipper::expression::` is complete. No remaining references to the old namespace.

### 5.3 Test File Organization

Test files generally mirror the source tree well:
- `tests/expression/unary/` for unary expression tests
- `tests/utils/` for utility tests

---

## 6. Test Coverage

### Current State

30 test executables, all passing. Total assertion count is substantial (hundreds).

### Coverage Gaps

| Area | Coverage | Notes |
|------|----------|-------|
| Sparse storage | Minimal | `SparseCoordinateAccessor` has basic tests but edge cases untested |
| Binary expression dangling | None | No tests for expression lifetime |
| Dynamic extents | Good | 50 test cases added in `test_dynamic_extents.cpp` |
| Error paths | Limited | Few tests for degenerate inputs |
| Large dimensions | None | All tests use small (2-5) dimensions |
| Form operations | Basic | `FormBase::operator*` unimplemented path untested |

### Recently Added Tests

| Test File | Tests | Assertions |
|-----------|-------|------------|
| `tests/utils/test_power_method.cpp` | 10 | 29 |
| `tests/expression/unary/anti_slice.cpp` | 7 | 30 |
| `tests/expression/unary/partial_reduction.cpp` | Rewritten | Fixed defunct API usage |
| `tests/expression/test_dynamic_extents.cpp` | 50 | 247 |

---

## 7. What We Fixed / Implemented

### Bugs Fixed
1. **`storage/StlStorageInfo.hpp:86`** -- Added missing `return v;`
2. **`utils/extents/is_cubic.hpp`** -- Removed dead code, added `return std::nullopt;`
3. **`tests/expression/unary/partial_reduction.cpp`** -- Complete rewrite from defunct `views::` API to current `expression::` API; fixed MDArray initialization bug

### New Implementations
4. **`utils/eigen/power_method.hpp`** -- Power iteration eigenvalue solver
   - Returns `std::expected<PowerMethodResult, PowerMethodError>`
   - Two overloads: random default initial guess + user-provided `concepts::Vector` guess
   - Constrained to `concepts::Matrix`
5. **`expression/unary/AntiSlice.hpp`** -- Dimension embedding expression
   - Writable (assignable)
   - Inserts a new dimension at a specified position
   - E.g., `Vector<3>` -> `Matrix<3,1>`

### Deleted
6. **`utils/solvers/inverse.hpp`** and empty `solvers/` directory

### Expression Storage Refactoring (Dangling Reference Fix)
7. **`detail/ExpressionStorage.hpp`** -- `expression_storage_t<T>` utility: stores by reference when `T` is lvalue ref, by value otherwise
8. **`expression/reductions/ReductionBase.hpp`** -- New CRTP base class for all 9 reduction classes
9. **`expression/binary/BinaryExpressionBase.hpp`** -- Refactored to use `expression_storage_t` with forwarding constructor
10. **`expression/unary/UnaryExpressionBase.hpp`** -- Refactored to use `expression_storage_t` with forwarding constructor
11. All 9 reduction, 8 binary, and ~15 unary expression classes updated to pass ref-qualified child types

### Code Quality Fixes
12. **`expression/nullary/StlMDArray.hpp`** -- Added missing `#pragma once`
13. **`detail/Features.hpp`** -- Fixed include guard typo (FEATUERS -> FEATURES)
14. **`expression/nullary/NullaryExpressionBase.hpp`** -- Removed unused `<spdlog/spdlog.h>` include
15. **`FormBase.hpp`**, **`expression/unary/Slice.hpp`** -- Replaced `assert(false)` with `std::unreachable()`
16. **`storage/SparseCoordinateAccessor.hpp`** -- Removed UB `namespace std` overloads; fixed double-slash in include path
17. **`utils/extents/extents_formatter.hpp`** -- Replaced UB `namespace std` `format_as` with proper `fmt::formatter` specialization; removed dead code
18. **`tests/storage/sparse_compressed_data.cpp`** -- Fixed typos in comments
19. **Renamed utility files to snake_case:** `maxCoeff.hpp` -> `max_coeff.hpp`, `minCoeff.hpp` -> `min_coeff.hpp`, `meanCoeff.hpp` -> `mean_coeff.hpp`

### Dynamic Extents Tests
20. **`tests/expression/test_dynamic_extents.cpp`** -- Added 50 test cases (247 assertions) covering dynamic-extent equivalents of static-extent operations: transpose, determinant, vector head/tail/segment, matrix row/col/block slicing, diagonal access/assignment, colwise/rowwise norm, unit vectors, homogeneous coordinates, custom unary_expr, reshape, scalar arithmetic, normalized. Also documented a library limitation where compound assignment operators (`+=`, `-=`, `*=`, `/=`) fail because `ZipperBase::operator+=` uses `*this + other` where `*this` has type `ZipperBase&` which doesn't satisfy the `Zipper` concept.

### Deducing This Refactoring (C++23)
21. **CRTP `derived()` methods refactored** -- Replaced all 6 paired const/non-const `derived()` overloads with a single deducing-this method using C++23 explicit object parameters:
    - `expression/ExpressionBase.hpp`
    - `expression/nullary/NullaryExpressionBase.hpp`
    - `expression/unary/UnaryExpressionBase.hpp`
    - `expression/binary/BinaryExpressionBase.hpp`
    - `expression/DynamicSizedExpressionBase.hpp`
    - `ZipperBase.hpp` (template-template `DerivedT<Expression>` pattern)

### MDArray Initialization Trap Fix
22. **`ZipperBase.hpp`** -- Constrained the variadic fallback constructor `ZipperBase(Args&&...)` to reject single `Expression` arguments when `Expression` is a non-reference (value) type. This prevents expressions like `Identity<double,3,3>` from slicing to their `extents<...>` base class and silently zero-initializing an MDArray. The `MDArray(const extents_type&)` constructor was already `explicit`, but the `ZipperBase` forwarding constructor bypassed that protection.

### ODR Violation Fix
23. **`include/zipper/detail/assert.hpp`** (new) -- Created `ZIPPER_ASSERT` macro controlled by `ZIPPER_DEBUG` (independent of `NDEBUG`) to eliminate ODR violations from `assert()` in header-defined templates. Replaced all 30 `assert()` calls across 16 header files:
    - `expression/ExpressionBase.hxx` (2) -- bounds checks in `access_index_pack`/`const_access_index_pack`
    - `expression/nullary/LinearLayoutExpression.hpp` (1) -- bounds check in `get_index`
    - `storage/SparseCoordinateAccessor.hpp` (4) -- iterator/find checks
    - `expression/unary/Reshape.hpp` (1) -- size preservation check
    - `expression/unary/Cofactor.hpp` (1) -- square matrix check
    - `expression/unary/Diagonal.hpp` (1) -- rank argument check
    - `expression/binary/FormTensorProduct.hpp` (1) -- pointer validity check
    - `expression/binary/MatrixVectorProduct.hpp` (2) -- dimension compatibility + rank argument
    - `expression/binary/MatrixProduct.hpp` (1) -- dimension compatibility
    - `expression/binary/CrossProduct.hpp` (3) -- dimension checks
    - `VectorBase.hpp` (1) -- initializer_list size check
    - `storage/StlStorageInfo.hpp` (2) -- static extent size checks
    - `Matrix.hpp` (6) -- initializer_list and constructor size checks
    - `Vector.hpp` (2) -- constructor size checks
    - `expression/reductions/Determinant.hpp` (1) -- square matrix check
    - `expression/detail/AssignHelper.hpp` (1) -- shape compatibility check
    - Also removed unused `#include <cassert>` from `utils/eigen/power_method.hpp`

---

## 8. Recommendations (Prioritized)

### Critical -- ALL FIXED
1. ~~**Fix dangling references in BinaryExpressionBase**~~ -- Done: `expression_storage_t` refactor across all expression families.
2. ~~**Fix reduction dangling references**~~ -- Done: new `ReductionBase` with `expression_storage_t`.
3. ~~**Prevent MDArray initialization trap**~~ -- Done: constrained `ZipperBase` variadic fallback to reject single `Expression` arguments when `Expression` is a value type.

### High -- ALL FIXED
4. ~~**Replace `assert(false)` with proper error handling**~~ -- Done: replaced with `std::unreachable()` in `FormBase.hpp` and `Slice.hpp`.
5. ~~**Remove UB in namespace std**~~ -- Done: removed overloads in `SparseCoordinateAccessor.hpp`, rewrote `extents_formatter.hpp` with `fmt::formatter` specialization.
6. ~~**Remove unused spdlog include**~~ -- Done: removed from `NullaryExpressionBase.hpp`.

### Medium -- ALL FIXED
7. ~~**Fix include guard typo**~~ -- Done: `Features.hpp` FEATUERS -> FEATURES.
8. ~~**Add missing include guard**~~ -- Done: `StlMDArray.hpp` now has `#pragma once`.
9. ~~**Consolidate pragma suppressions**~~ -- Done: consolidated 6 identical `-Weffc++` pragma blocks in `ZipperBase.hpp` into a single push/pop block; removed duplicate `-Wextra-semi` in `types.hpp`.
10. ~~**Clean up orphaned files**~~ -- Done: deleted 6 orphaned files that were never included and contained old/broken code: `Addition.hpp`, `CoeffProduct.hpp`, `ExpressionTypeHelper.hpp`, `DynamicSizedExpressionBase.hpp`, `DiagonalTensor.hpp`, `SparseLayoutPolicy.hpp`. (2 others — `Contraction.hpp`, `pivot_utils.hpp` — were already deleted.)

### Low -- ALL DONE
11. ~~**Add tests for sparse storage edge cases**~~ -- Done: added 35 new test cases across `sparse_coordinate_accessor.cpp` and `sparse_compressed_data.cpp` (total 55 test cases, 1488 assertions in storage suite). Covers: uncompressed find path, duplicate cancellation, negative values, integer/float value types, copy/move semantics, empty accessor operations, compress idempotency, boundary indices, large extents, non-square dynamic extents, dynamic rank-1/rank-3, iterator arithmetic, `SparseCompressedData` empty/single/many-element cases, out-of-order insert error path, and `to_sparse_compressed_data` conversions (empty, rank-1, rank-3, duplicates, dynamic extents).
12. ~~**Add tests with dynamic extents**~~ -- Done: 50 test cases (247 assertions) in `test_dynamic_extents.cpp`.
13. ~~**Adopt C++23/26 features**~~ -- Done: (a) deducing this for all 6 CRTP `derived()` methods; (b) `static operator()` for 3 functor structs (`Abs`, `ScalarPower`, `min`/`max`); (c) `std::print`/`std::println` migration across 10 test/example files (~40 call sites). Library headers still use `fmt::format`/`fmt::join` (no `std::` equivalent for `fmt::join`).
14. ~~**Fix ODR violations from `assert()` in headers**~~ -- Done: replaced all 30 `assert()` calls with `ZIPPER_ASSERT()` macro (controlled by `ZIPPER_DEBUG`, independent of `NDEBUG`) across 16 header files.
