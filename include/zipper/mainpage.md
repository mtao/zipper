# Zipper {#mainpage}

**Zipper** is a C++26 header-only tensor processing library inspired by
[Eigen](https://eigen.tuxfamily.org/) and
[std::mdspan](https://en.cppreference.com/w/cpp/container/mdspan).
It provides lightweight, clean syntax for linear algebra and tensor algebra
using expression templates and CRTP (Curiously Recurring Template Pattern).

## Quick Start

### Creating Vectors and Matrices

```cpp
#include <zipper/Vector.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Form.hpp>

using namespace zipper;

// Static-sized types (dimensions known at compile time)
Vector<double, 3> v({1.0, 2.0, 3.0});
Matrix<double, 3, 3> A({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}});
Form<double, 3> f({1.0, 2.0, 3.0});

// Dynamic-sized types (dimensions chosen at runtime)
Vector<double, dynamic_extent> w(100);
Matrix<double, dynamic_extent, dynamic_extent> B(4, 4);
```

### Arithmetic

```cpp
// Coefficient-wise operations
auto sum = v + w;
auto scaled = 2.0 * v;

// Matrix-vector product
Vector<double, 3> y = A * v;

// Dot product
double d = v.dot(w);

// Form-vector contraction (produces scalar)
double s = f * v;   // inner product

// Vector-form product (produces matrix)
Matrix<double, 3, 3> outer = v * f;   // outer product
```

### Solving Linear Systems

```cpp
#include <zipper/utils/solver/conjugate_gradient.hpp>
#include <zipper/utils/solver/triangular_solve.hpp>
#include <zipper/utils/decomposition/qr.hpp>
#include <zipper/utils/inverse.hpp>

using namespace zipper::utils;

// Conjugate Gradient (for symmetric positive definite A)
auto result = solver::conjugate_gradient(A, b);
if (result) {
    Vector<double, 3> x = result->x;
    double residual = result->residual_norm;
    int iters = result->iterations;
}

// Triangular solve (direct, O(n^2))
auto L = triangular_view<TriangularMode::Lower>(A);
auto x = solver::triangular_solve(L, b);

// QR solve (for square or overdetermined systems)
auto x = decomposition::qr_solve(A, b);

// Matrix inverse
Matrix<double, 3, 3> Ainv = inverse(A);
```

## Architecture

Zipper is built on a layered architecture:

| Layer | Purpose |
|-------|---------|
| @ref user_types "User-Facing Types" | `Matrix`, `Vector`, `Form`, `Tensor` -- owning types with semantic operations |
| **ZipperBase / ArrayBase** | CRTP bases providing common interface (assign, arithmetic) |
| @ref expressions "Expression System" | Lazy expression templates evaluated on assignment |
| @ref storage "Storage" | Dense (`StlStorage`) and sparse (`SparseCoordinate`) backends |
| @ref concepts "Concepts" | Compile-time constraints (`concepts::Matrix`, `concepts::Vector`, etc.) |

### Expression Templates

All arithmetic operations build lazy expression trees that are evaluated only
when assigned to an owning type. Expressions are organized as:

- @ref expressions_nullary "Nullary" -- leaf nodes: MDArray, Identity, Unit, Constant
- @ref expressions_unary "Unary" -- single child: Transpose, Slice, TriangularView
- @ref expressions_binary "Binary" -- two children: arithmetic, MatrixProduct
- @ref expressions_reductions "Reductions" -- collapse dimensions: Determinant, Trace

### Zero-Aware Sparsity Protocol

The @ref sparsity "sparsity protocol" enables expressions to declare which
elements are structurally zero. Operations like `MatrixProduct` and
`MatrixVectorProduct` use this information to skip zero contributions,
providing significant speedups for triangular and structured computations.

Participating expressions:
- `TriangularView` -- contiguous ranges from triangular structure
- `Identity` -- single-index ranges (one nonzero per row/column)
- `Unit` -- single-index range (one nonzero element)
- `ZeroAwareOperation` -- propagates ranges through addition/subtraction

## Modules

| Module | Description |
|--------|-------------|
| @ref user_types | Matrix, Vector, Form, Array, Tensor |
| @ref expressions | Expression template hierarchy |
| @ref sparsity | Nonzero tracking and zero-aware operations |
| @ref solvers | Iterative solvers (CG, GMRES, BiCGSTAB, Gauss-Seidel, triangular) |
| @ref decompositions | QR decomposition, matrix inverse |
| @ref storage | Dense and sparse storage backends |
| @ref concepts | C++ concept definitions |

## Choosing a Solver

| Problem type | Recommended solver |
|--------------|--------------------|
| Triangular system | `triangular_solve` (direct, O(n^2)) |
| Symmetric positive definite | `conjugate_gradient` |
| Diagonally dominant | `gauss_seidel` |
| General non-symmetric | `gmres` or `bicgstab` |
| Overdetermined (least squares) | `qr_solve` |

## Requirements

- **C++26** compiler (GCC 14+ or recent Clang)
- **Meson** build system (for tests and examples)
- Header-only: just add `include/` to your include path
