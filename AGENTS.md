# AGENTS.md

Instructions for AI coding agents working on the Zipper codebase.

---

## Project Overview

**Zipper** is a C++26 header-only tensor processing library inspired by
[Eigen](https://eigen.tuxfamily.org/) and
[std::mdspan](https://en.cppreference.com/w/cpp/container/mdspan). It provides
lightweight, clean syntax for linear algebra and tensor algebra using expression
templates and CRTP (Curiously Recurring Template Pattern).

- **Author:** Michael Tao
- **License:** MIT
- **Language standard:** C++26
- **Build system:** Meson + Ninja
- **Test framework:** Catch2 v3.8.1
- **Header-only:** All library code lives in `include/zipper/`

---

## Quick Reference

### Build

```bash
# Configure (with tests enabled)
meson setup build/ -Dtesting=true

# Build
meson compile -C build/

# Build (with subproject fallback, matches CI)
meson setup build/ --wrap-mode=forcefallback -Dtesting=true
meson compile -C build/
```

### Test

```bash
meson test -C build/ -v
```

### Build with Conan (alternative)

```bash
conan install . --output-folder=build/conan --build=missing
pushd build
meson setup --native-file conan/conan_meson_native.ini .. . -Dtesting=true
ninja
popd
```

### Existing Build Directories

The developer may have pre-configured build directories on disk:
`build-debug/`, `build-debug-clang/`, `build-release/`, `builddir/`. Check
which exist before creating new ones. Prefer reusing an existing build
directory when possible (run `meson compile -C <dir>` and `meson test -C <dir> -v`).

### Generate Documentation

```bash
ninja -C build/ docs   # requires doxygen
```

---

## Dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| `fmt` | 11.2.0 | Formatting (fallback until `std::format` ranges work) |
| `mdspan` | 0.6.0 | Multi-dimensional span (Kokkos, used as Meson subproject) |
| `catch2` | 3.8.1 | Testing framework |
| `range-v3` | 0.12.0 | Range utilities (Conan only, commented out in Meson) |

Dependencies are resolved via Meson subprojects (`subprojects/`) or Conan
(`conanfile.py`). CI uses `--wrap-mode=forcefallback` to fetch all via
subprojects.

### Build Options (`meson_options.txt`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `testing` | boolean | `false` | Enable unit testing |
| `examples` | boolean | `false` | Enable example targets |
| `use_mdspan_subproject` | boolean | `true` | Use mdspan from subproject |

### Compiler Requirements

- **GCC 14+** or **Clang** with C++26 support
- CI matrix: Linux (GCC 14, ubuntu-24.04) + macOS (Clang, macos-15)
- Both `debug` and `release` build types are tested in CI

### Compile-Time Feature Detection

The `meson.build` probes the compiler and sets these defines automatically:

| Define | Condition |
|--------|-----------|
| `-DZIPPER_FMT_OVERRIDES_DISABLED` | `std::format(ranges)` does not compile |
| `-DZIPPER_DONT_USE_ALIAS_CTAD` | Alias CTAD not supported |
| `-DZIPPER_DEBUG` | Enables `ZIPPER_ASSERT` runtime checks (set manually) |

---

## Directory Structure

```
zipper/
├── include/zipper/           # All library headers (header-only)
│   ├── concepts/             # C++ concept definitions (16 files)
│   ├── detail/               # Internal utilities (17+ files)
│   ├── expression/           # Expression template hierarchy
│   │   ├── nullary/          # Leaf expressions (MDArray, Identity, Constant, etc.)
│   │   ├── unary/            # Single-operand (Slice, Transpose, Diagonal, etc.)
│   │   ├── binary/           # Two-operand (MatrixProduct, CrossProduct, etc.)
│   │   ├── reductions/       # Reductions (Determinant, Trace, Norm, etc.)
│   │   ├── concepts/         # Expression-specific concepts
│   │   └── detail/           # ExpressionTraits, AssignHelper
│   ├── storage/              # Storage backends (dense, sparse, span)
│   │   ├── concepts/         # Storage concepts (Data, DenseData, etc.)
│   │   └── detail/           # Sparse compressed internals
│   ├── utils/                # Higher-level algorithms
│   │   ├── solver/           # Iterative solvers (CG, BiCGSTAB, GMRES, etc.)
│   │   ├── krylov/           # Power method, Arnoldi, Lanczos
│   │   ├── decomposition/    # QR decomposition
│   │   ├── orthogonalization/ # Gram-Schmidt
│   │   └── extents/          # Extent manipulation utilities
│   ├── {Array,Vector,Matrix,Form,Tensor,Container}.hpp       # Owning types
│   ├── {Array,Vector,Matrix,Form,Tensor,Container}Base.hpp   # CRTP wrapper bases
│   ├── {Array,Vector,Matrix,Form,Tensor,Container}Base.hxx   # Implementation files
│   ├── ZipperBase.hpp        # Core CRTP base for all semantic wrappers
│   ├── types.hpp             # Fundamental type aliases
│   └── as.hpp                # Type conversion utilities
├── tests/                    # Test suite (~65 .cpp files, ~30 executables)
│   ├── concepts/             # Concept tests
│   ├── detail/               # Detail utility tests
│   ├── expression/           # Expression tests (mirrors include/ structure)
│   │   ├── binary/
│   │   ├── nullary/
│   │   ├── unary/
│   │   └── reductions/
│   ├── integration/          # Integration tests
│   ├── storage/              # Storage tests (dense + sparse)
│   └── utils/                # Utility/algorithm tests
├── examples/                 # Example programs
├── subprojects/              # Meson subprojects (mdspan, etc.)
├── .github/workflows/        # CI: continuous_integration.yml, docs.yml
├── meson.build               # Root build definition
├── meson_options.txt         # Build options
├── conanfile.py              # Conan package manager recipe
├── Doxyfile                  # Doxygen configuration
├── ANALYSIS.md               # Architecture analysis, bug log, fix history
└── README.md                 # Project overview
```

---

## Architecture

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
│  ├── binary/   (CWiseBinary, MatrixProduct, etc.)    │
│  └── reductions/ (Sum, Norm, Determinant, Trace)     │
├─────────────────────────────────────────────────────┤
│  Storage layer                                       │
│  StlStorage, SparseCoordinate, layout policies       │
├─────────────────────────────────────────────────────┤
│  Concepts + Traits                                   │
│  concepts::Matrix, concepts::Vector, etc.            │
│  ExpressionTraits<T>, AccessFeatures, ShapeFeatures  │
└─────────────────────────────────────────────────────┘
```

### Key Design Patterns

- **CRTP expression templates:** `ExpressionBase<Derived>` is the root.
  Unary expressions inherit `UnaryExpressionBase<Derived, ChildExpr>`.
  Binary expressions use `BinaryExpressionBase<Derived, LHS, RHS>`.
  Reductions use `ReductionBase<Derived, ChildExpr>`.

- **Traits-based dispatch:** `ExpressionTraits<T>` specializations define
  `value_type`, `extents_type`, `AccessFeatures`, `ShapeFeatures`.
  **Important:** `ExpressionTraits<T&>` does NOT exist -- always use
  `std::remove_reference_t` before lookup.

- **Concept hierarchy:** `concepts::Matrix`, `concepts::Vector`,
  `concepts::Form`, `concepts::Zipper` are top-level. `concepts::Expression`
  and `RankedExpression<T,N>` only match raw expression types (like `MDArray`),
  NOT the wrapper types (like `Vector`, `Matrix`).

- **Expression storage (reference safety):** `expression_storage_t<T>` from
  `detail/ExpressionStorage.hpp` stores lvalue refs by reference and rvalues by
  value. This prevents dangling references in expression trees -- a key
  improvement over Eigen's approach.

- **`.hpp`/`.hxx` split:** Interface declarations in `.hpp`, CRTP
  implementation bodies in `.hxx` (7 `.hxx` files for the main `*Base` classes).

### Semantic Type System

The library separates storage/expression from mathematical semantics:

| Type | Rank | Semantics |
|------|------|-----------|
| `Array<T,...>` / `ArrayBase<Expr>` | Any | Dense array, coefficient-wise ops only |
| `Vector<T,N>` / `VectorBase<Expr>` | 1 | Column vector |
| `Form<T,N>` / `FormBase<Expr>` | 1 | Row vector / 1-form (`Form * Vector` = scalar) |
| `Matrix<T,R,C>` / `MatrixBase<Expr>` | 2 | Matrix (row/col/block access, determinant, etc.) |
| `Tensor<T,...>` / `TensorBase<Expr>` | Any | General tensor |
| `Container<T,...>` / `ContainerBase<Expr>` | Any | Generic container (currently disabled) |

Owning types (e.g., `Vector<double,3>`) own their data via `MDArray`. `*Base`
types wrap an expression by reference and impose semantics without copying.

### C++26 Features in Use

| Feature | Usage |
|---------|-------|
| Pack indexing (`args...[K]`) | Guarded by `__cpp_pack_indexing` |
| `consteval` functions | 76+ sites throughout |
| Concepts | Core architecture (43+ concepts) |
| Deducing this (C++23) | All CRTP `derived()` methods |
| `std::expected` (C++23) | Power method solver error handling |
| `std::unreachable()` (C++23) | Replaces `assert(false)` in impossible branches |
| `static operator()` (C++23) | 3 functor structs (`Abs`, `ScalarPower`, min/max) |

---

## Code Conventions

### File Naming

- **Expression headers:** PascalCase (`Slice.hpp`, `MatrixProduct.hpp`,
  `AntiSlice.hpp`)
- **Utility headers:** snake_case (`power_method.hpp`, `max_coeff.hpp`,
  `is_cubic.hpp`)
- **Implementation files:** `.hxx` extension for CRTP method bodies
  (e.g., `VectorBase.hxx` implements `VectorBase.hpp`)
- **Test files:** snake_case, mirroring the source tree structure

### Include Guards

Use `#pragma once` for all new headers.

### Assertions

Use `ZIPPER_ASSERT(expr)` (from `include/zipper/detail/assert.hpp`) instead of
`assert()`. This macro is controlled by `-DZIPPER_DEBUG` (independent of
`NDEBUG`) to avoid ODR violations in header-only template code. When
`ZIPPER_DEBUG` is not defined, it compiles to a no-op with zero overhead.

**Never use `assert()` in headers.** It causes ODR violations when different
translation units define `NDEBUG` differently.

### Error Handling

- Use `std::unreachable()` for code paths that should never be reached (not
  `assert(false)`).
- Use `std::expected` for functions that can fail (see `power_method.hpp` for
  the established pattern).

### Compiler Warnings

- Default: `-Wall -Wextra -Wpedantic` (warning_level=3 in `meson.build`)
- Tests additionally suppress: `-Wno-padded`, `-Wno-float-equal`
- Avoid adding new `#pragma GCC diagnostic` blocks where possible; prefer
  fixing the warning or isolating external includes behind wrapper headers.

### Namespace

All library code is in the `zipper` namespace. Sub-namespaces:
- `zipper::expression::` -- expression template types
- `zipper::expression::nullary::`, `::unary::`, `::binary::` -- expression categories
- `zipper::concepts::` -- C++ concept definitions
- `zipper::storage::` -- storage backends
- `zipper::detail::` -- internal implementation details

**Never add overloads or specializations to `namespace std`.** Use `fmt::`
formatter specializations or ADL-based customization points instead.

---

## Adding New Code

### Adding a New Expression

1. Create the header in the appropriate subdirectory under
   `include/zipper/expression/` (nullary, unary, binary, or reductions).
2. Inherit from the correct base: `NullaryExpressionBase`, `UnaryExpressionBase`,
   `BinaryExpressionBase`, or `ReductionBase`.
3. Specialize `ExpressionTraits<YourType>` to define `value_type`,
   `extents_type`, `AccessFeatures`, and `ShapeFeatures`.
4. Use `expression_storage_t<T>` for storing child expressions to ensure
   correct reference/value semantics.
5. Add tests in the corresponding `tests/expression/` subdirectory.
6. Register the test in the relevant `tests/.../meson.build`.

### Adding a New Utility / Algorithm

1. Create the header in `include/zipper/utils/` (or an appropriate subdirectory
   like `solver/`, `krylov/`, `decomposition/`).
2. Use snake_case for file naming.
3. Constrain template parameters with existing concepts (`concepts::Matrix`,
   `concepts::Vector`, etc.).
4. Add tests in `tests/utils/`.
5. Register the test in `tests/utils/meson.build`.

### Adding a New Test

1. Create the `.cpp` file in the appropriate `tests/` subdirectory.
2. Include `<catch2/catch_test_macros.hpp>` (via `catch_include.hpp`).
3. Register it in the corresponding `meson.build`:
   - Either add it to an existing multi-source test executable, or
   - Create a new executable + `test()` call.
4. Verify: `meson compile -C build/ && meson test -C build/ -v`

---

## Testing Guidelines

- **All 30+ test executables must pass** before submitting changes.
- Tests are organized to mirror the `include/zipper/` directory structure.
- Use Catch2's `TEST_CASE` and `SECTION` macros.
- Test both static and dynamic extents where applicable.
- The `Container` tests are currently disabled (need porting to `ZipperBase`
  API) -- see `tests/meson.build:35-39`.
- Run `meson test -C build/ -v` for verbose output with test names.

### Known Limitation

Compound assignment operators (`+=`, `-=`, `*=`, `/=`) fail on dynamic-extent
types because `ZipperBase::operator+=` constructs `*this + other` where `*this`
has type `ZipperBase&`, which doesn't satisfy the `Zipper` concept. This is a
known library limitation, not a test bug.

---

## CI/CD

### Continuous Integration (`.github/workflows/continuous_integration.yml`)

- **Triggers:** Push to `main`, all pull requests
- **Matrix:** 4 jobs (2 platforms x 2 build types)
  - Linux: GCC 14, ubuntu-24.04
  - macOS: Clang, macos-15
  - Build types: debug, release
- **Meson version:** 1.7.2
- **Caching:** ccache (500MB), subproject downloads
- **Flags:** `--wrap-mode=forcefallback` (uses subprojects for all dependencies)

### Documentation (`.github/workflows/docs.yml`)

- Deploys Doxygen-generated HTML to GitHub Pages on push to `main`.

---

## Important Caveats for Agents

1. **This is a header-only library.** There are no `.cpp` source files to
   compile for the library itself -- only for tests and examples.

2. **C++26 is required.** Do not introduce code that lowers the language
   standard. Use modern features (concepts, consteval, deducing this, etc.).

3. **ExpressionTraits must be specialized** for every new expression type.
   Forgetting this will cause cryptic compile errors. Always
   `remove_reference_t` before looking up traits.

4. **Reference safety is critical.** Always use `expression_storage_t<T>` when
   storing child expressions. Passing `const A&` as the template parameter
   stores by reference; passing `A` stores by value. Getting this wrong causes
   dangling references.

5. **Concept hierarchy matters.** `concepts::Expression` does NOT match wrapper
   types like `Vector` or `Matrix`. Use `concepts::Vector`, `concepts::Matrix`,
   or `concepts::Zipper` for the wrappers. `QualifiedExpression` handles
   ref-qualified types.

6. **No `assert()` in headers.** Use `ZIPPER_ASSERT()` exclusively.

7. **No `namespace std` pollution.** Never add overloads or specializations in
   `namespace std`.

8. **Orphaned files exist.** Several stub/incomplete files are intentionally
   kept but not included anywhere (listed in `ANALYSIS.md` section 4). Do not
   try to fix or include them unless specifically asked.

9. **The `ANALYSIS.md` file** contains detailed architecture analysis, a
   complete bug fix history, and prioritized recommendations. Read it before
   making architectural changes.

10. **Build directories vary.** The developer uses multiple build directories
    (`build-debug/`, `build-release/`, `build-debug-clang/`, `builddir/`).
    Check what exists before creating new ones.
