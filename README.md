
# Zipper

A zippy C++26 tensor processing library. Heavily inspired by [Eigen](https://eigen.tuxfamily.org/) and [mdspan](https://en.cppreference.com/w/cpp/container/mdspan), this library-in-development intends to provide a light-weight clean syntax for linear algebra and tensor algebra while also providing convenient backends to transform between different data representations.

## Principles

The underlying concept in Zipper is an _expression_, which represents an underlying map from integer indices to values. When an expression represents a (multi-dimensional) array of objects, the semantics of vectors (rank-1), matrices (rank-2), tensors, or forms are induced by wrapping that expression in a typed container.

More concretely, we can take unit vectors as

```cpp
#include <zipper/expression/nullary/Unit.hpp>

// unit vector e_1 = {0,1,0}
auto e1 = zipper::expression::nullary::unit_vector<double, 3>(1);
```

This expression can be used with different semantic wrappers:

```cpp
#include <zipper/VectorBase.hpp>
#include <zipper/FormBase.hpp>

// a vector wrapping the unit expression
zipper::VectorBase x(e1);

// a 1-form (row-vector) wrapping the same expression
zipper::FormBase u(e1);
```

With these semantic wrappers we can do semantic-specific operations:

```cpp
// form * vector contracts to a scalar (dot product)
double res = u * x;

// vector * form produces a rank-2 outer product
auto outer = x * u;

// a vector holding a lazy expression for (2 * x)
zipper::VectorBase x2 = 2 * x;

// materialise into an owning vector
zipper::Vector x2_ = x2;
```

### Expressions and Value Categories

Those who are used to Eigen will recognize this structure of building expression templates using CRTP. The main difference here is that the semantics are kept separate from the inheritance hierarchy and instead class membership is used. This has the subtle advantage of letting us be careful about value categories.

The key mechanism is `expression_storage_t<T>`: when an expression node
captures a child operand, lvalue references are stored by reference while
rvalue temporaries are moved in and stored by value. This means intermediate
results that are temporaries are automatically owned by the expression tree,
while named variables are cheaply referenced. The choice is made per-operand at
compile time based on the value category of the argument.

For example:

```cpp
zipper::Vector<double,4> x = {0,1,2,3};
auto p = x + 3 * zipper::Vector<double,4>({2,3,4,5});
```

results in `p` being stored as something like

```
struct Addition {
    zipper::Vector<double,4>& lhs; // = x (lvalue, stored by reference)
    struct ScalarProduct {
        double lhs;                     // = 3
        zipper::Vector<double,4> rhs;   // = {2,3,4,5} (rvalue, owned by value)
    } rhs;                              // (rvalue, owned by value)
};
```

Every rvalue operand in the expression is stored by value rather than by
reference, so it survives the end of the full-expression. Named variables like
`x` are stored by reference (cheap, zero-copy). In Eigen, from my experience,
the internal ScalarProduct would be stored as a reference, so the temporary
object would disappear at the end of the line and the expression would
therefore point to invalid memory.

### Ownership, Views, and Returning Expressions

**Expressions are safe by default.** As described above, `expression_storage_t`
automatically stores rvalue child expressions by value and lvalue operands by
reference. This means that composing expressions — even ones built from
temporary views — is inherently safe:

```cpp
zipper::Matrix<double, 3, 3> S{{1,2,3},{4,5,6},{7,8,9}};

// Safe: the + operator captures both temporary row views by value.
// The resulting expression owns the Slice nodes; S's data is still live.
auto v = S.row(std::integral_constant<zipper::index_type, 0>{})
       + S.row(std::integral_constant<zipper::index_type, 1>{});
// v(0) == 5.0, v(1) == 7.0, v(2) == 9.0
```

Both `S.row(0)` and `S.row(1)` are rvalue temporaries, so the addition
expression moves them in by value. The Slice nodes survive inside the
expression tree, and their references back to `S`'s storage remain valid as
long as `S` is alive. **This is the core safety property that distinguishes
Zipper from Eigen**, where the equivalent code silently dangles.

#### The View Limitation

The one case where C++ works against us is capturing a bare view with `auto`:

```cpp
auto s = S.col(zipper::index_type(0));   // compiles (prvalue, no copy)
auto s2 = s;                             // ERROR: copy constructor deleted
```

`S.col(j)` returns a view that holds a reference to `S`. Because that
reference is tracked at the type level via `stores_references`, the view
inherits `NonReturnable`, which deletes the copy constructor. This is
fundamentally a C++ language limitation — there is no way for a member function
to express "my return value borrows from `*this`" in the type system. Rather
than silently allowing these references to dangle (as Eigen does), Zipper makes
the error a compile-time failure and requires explicit opt-in to suppress it.

Views still work perfectly in local scope and in expression chains. Assignment
through temporary views works directly:

```cpp
// No unsafe() needed -- the temporary lives long enough
M.col(zipper::index_type(1)) = zipper::Vector<double, 3>{10, 20, 30};

// Live mutable view -- no copy, no allocation
auto r = M.row(std::integral_constant<zipper::index_type, 0>{});
r(1) = 99.0;          // writes through to M(0,1)
M(0, 2) = 42.0;       // visible through r(2)
```

#### Escape Hatches

When you do need to materialize, snapshot, or pass around expressions, Zipper
provides three mechanisms, listed in order of preference:

##### `eval()` -- Materialize to a Concrete Type

`eval()` eagerly evaluates the entire expression tree into a concrete owning
type (`Vector`, `Matrix`, etc.). The result is fully independent of the
original data.

```cpp
zipper::Vector<double, 3> a{1, 2, 3}, b{10, 20, 30};

auto sum_lazy = a + b;     // lazy -- still references a and b
auto sum = (a + b).eval(); // materialized -- owns its data

a(0) = 999.0;
// sum_lazy(0) == 1009.0   -- sees the mutation
// sum(0)      == 11.0     -- independent copy
```

Use `eval()` when you want a concrete value type (e.g., to return from a
function, store in a container, or break a long expression chain).

##### `to_owned()` -- Deep-Copy the Expression Tree

`to_owned()` recursively deep-copies every node in the expression tree so that
no references remain, but preserves the lazy expression template structure.
The result has `stores_references == false` and is safe to copy, move, and
return.

```cpp
auto expr = 2.0 * a + 3.0 * b;     // lazy, references a and b
auto owned = expr.to_owned();       // deep copy -- no references remain

static_assert(!decltype(owned)::stores_references);
a(0) = 0.0;
// owned(0) still == 32.0  -- fully independent
```

Use `to_owned()` when you want to snapshot a lazy expression without
materializing it (e.g., to store it for later evaluation).

##### `unsafe()` -- Copyable/Returnable Views (Caller Asserts Lifetime)

`unsafe()` wraps the expression in an `UnsafeRef` node that overrides
`stores_references` to `false`, making the result copyable and returnable. The
caller asserts that the referenced data will outlive all uses. This adds minor
verbosity, but the trade-off is intentional: rather than silently allowing
dangling references, Zipper requires explicit opt-in for potentially unsafe
lifetime escapes.

Note that `auto s = M.col(j)` works fine on its own (prvalue, guaranteed copy
elision). You only need `unsafe()` when you need to **copy** the view or
**return** it from a function:

```cpp
zipper::Matrix<double, 3, 3> M{{1,2,3},{4,5,6},{7,8,9}};

auto s = M.col(zipper::index_type(1));            // OK -- prvalue, no copy
// auto s2 = s;                                   // ERROR -- copy deleted
auto s2 = s.unsafe();                             // OK -- copyable view
s2(0) = 42.0;                                     // writes through to M(0,1)

// Returning from a function requires unsafe():
auto get_col = [&M](int j) {
    return M.col(zipper::index_type(j)).unsafe();  // moved into UnsafeRef
};
```

`unsafe()` is ref-qualified for safety:
- **On lvalues** (`&` / `const &`): the `UnsafeRef` stores a reference to the
  expression (lightweight, zero-copy).
- **On rvalues** (`&&`): the `UnsafeRef` moves the expression node in by
  value, so it survives the temporary. The owned node may still hold internal
  references (e.g., a `Slice` referencing the original matrix's storage).

```cpp
// Rvalue chaining -- Slice is moved into UnsafeRef, M's data is still live
auto c = M.col(zipper::index_type(0)).unsafe();

// Lvalue -- UnsafeRef holds a reference to the view in `row_view`
auto row_view = M.row(std::integral_constant<zipper::index_type, 0>{});
auto r = row_view.unsafe();
```

#### Summary

| Method | Result | Ownership | Use When |
|---|---|---|---|
| `eval()` | Concrete `Vector`/`Matrix`/etc. | Fully independent | You need a value type |
| `to_owned()` | Lazy expression tree (deep copy) | Fully independent | You want to snapshot a lazy expression |
| `unsafe()` | Returnable view wrapper | References original data | You need to copy or return a view |

## DataArray

`DataArray<T, N...>` is the plain-data storage type in Zipper. It owns a
dense multidimensional array with no arithmetic semantics — to do math, wrap
it in an algebraic type (`.as_vector()`, `.as_matrix()`, etc.) or use
`as_array()` for coefficient-wise operations.

```cpp
#include <zipper/DataArray.hpp>

// Static-extent 1D array
zipper::DataArray<double, 4> a;
a(0) = 1.0; a(1) = 2.0; a(2) = 3.0; a(3) = 4.0;

// 2D array
zipper::DataArray<int, 2, 3> grid;
grid(0, 0) = 1; grid(1, 2) = 6;

// Dynamic-extent array
zipper::detail::DataArray_<double, zipper::dextents<1>> dyn(5);
```

### DataArray utilities

```cpp
// fill() -- set all elements to a value
a.fill(42.0);

// zero() -- static factory returning a zero-initialised array
auto z = zipper::DataArray<double, 3>::zero();

// reshape() -- reinterpret as different extents (copies data)
auto mat = a.reshape(zipper::extents<2, 2>{});  // 4-vector -> 2x2
```

DataArray also provides `data()`, `begin()`/`end()`, `as_span()`, `eval()`,
slicing, swizzling (transpose), `cast<U>()`, and lexicographic comparison
(`operator==`, `operator<=>`).

## Tensor Contraction

Zipper provides free functions for tensor product and contraction operations:

```cpp
#include <zipper/Tensor.hpp>

zipper::Tensor<double, 2, 3> A;
zipper::Tensor<double, 3, 4> B;

// Outer (tensor) product: produces a rank-4 tensor (2,3,3,4)
auto tp = zipper::tensor_product(A, B);

// Contract a pair of indices: contract<I,J>(tensor) traces over
// indices I and J, reducing rank by 2
auto c = zipper::contract<1, 2>(tp);  // (2,3,3,4) -> (2,4)

// Full contraction: fold-in-half sugar that contracts
// index 0 with N/2, 1 with N/2+1, etc., reducing to a scalar
zipper::Tensor<double, 3, 3> M;
double tr = zipper::full_contract(M);  // equivalent to trace
```

## Quaternions

`Quaternion<T>` provides quaternion algebra with scalar-first storage
`(w, x, y, z)`, matching the mathematical convention `q = w + xi + yj + zk`.

```cpp
#include <zipper/Quaternion.hpp>

zipper::Quaternion<double> q(1.0, 0.0, 0.0, 0.0);  // identity
zipper::Quaternion<double> r(0.0, 1.0, 0.0, 0.0);   // 180 degrees around x

auto product = q * r;          // Hamilton product (lazy expression)
auto conj = q.conjugate();     // q* = w - xi - yj - zk
auto inv = q.inverse();        // q* / |q|^2
double n = q.norm();           // |q|
auto u = q.normalized();       // unit quaternion
double d = q.dot(r);           // quaternion inner product

// Component access
double w = q.w();  // scalar
double x = q.x();  // i
double y = q.y();  // j
double z = q.z();  // k
```

The Hamilton product is implemented as a lazy binary expression
(`HamiltonProduct`), following the same expression template architecture as
matrix and vector operations.

## Sparse Types

Zipper provides sparse matrix and vector types in two storage formats:
coordinate (COO) for construction and mutation, and compressed (CSR/CSC) for
efficient arithmetic.

### COO (Coordinate) Format

```cpp
#include <zipper/COOMatrix.hpp>

// Build a sparse matrix by inserting entries
zipper::COOMatrix<double, 3, 3> A;
A.emplace(0, 0) = 4.0;
A.emplace(0, 1) = -1.0;
A.emplace(1, 0) = -1.0;
A.emplace(1, 1) = 4.0;
A.emplace(2, 2) = 3.0;

A.compress();  // sort and deduplicate
```

### Compressed Format (CSR / CSC)

```cpp
#include <zipper/CSRMatrix.hpp>
#include <zipper/CSMatrix.hpp>

// Convert COO to CSR
auto B = A.to_csr();

// CSMatrix is the unified type, parameterized by layout:
//   layout_right = CSR (row-compressed, default)
//   layout_left  = CSC (column-compressed)
auto csc = B.as_csc();    // convert to CSC
auto coo = B.to_coo();    // convert back to COO
```

Sparse types support `coeff()` for read access (returning 0 for missing
entries) and provide `index_set<D>()` for zero-aware expression optimizations
(e.g., skipping known-zero entries in matrix-vector products).

Sparse vectors are also available: `COOVector<T, N>` and `CSVector<T, N>`.

## Transforms

The `zipper::transform` namespace provides geometric transform types and
factory functions for spatial transformations, projections, and coordinate
mapping. All types follow right-handed coordinate conventions (OpenGL style).

### Transform Types

```cpp
#include <zipper/transform/transform.hpp>  // umbrella header
using namespace zipper::transform;

// General matrix-backed transform (4x4 for 3D)
AffineTransform<float, 3> xform;
auto lin = xform.linear();       // 3x3 view of upper-left block
auto t   = xform.translation();  // 3-vector view of last column
auto inv = xform.inverse();      // mode-aware inverse

// Specialised types (minimal storage)
Rotation<float, 3> R;          // 3x3 orthogonal matrix
Scaling<float, 3> S;           // 3-vector of scale factors
Translation<float, 3> T;      // 3-vector displacement
AxisAngleRotation<float> aa;   // angle + axis

// Composition via operator*
auto composed = T * R * S;  // modes promote automatically
```

`TransformMode` controls inverse computation: `Isometry` (transpose),
`Affine` (block inverse), `Projective` (general inverse). When composing
transforms, the result uses the least restrictive mode.

### Projections and View Matrices

```cpp
auto P = perspective(radians(45.0f), 16.0f/9.0f, 0.1f, 100.0f);
auto O = ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);
auto V = look_at(eye, center, up);
```

### Quaternion Transforms

```cpp
auto q = angle_axis(radians(90.0f), zipper::Vector<float, 3>({0, 1, 0}));
auto A = to_affine(q);              // quaternion -> AffineTransform
auto M = to_rotation_matrix(q);     // quaternion -> 3x3 rotation matrix
auto q2 = to_quaternion(A);         // affine -> quaternion
auto q3 = slerp(q, q2, 0.5f);      // spherical linear interpolation
```

See [`include/zipper/transform/README.md`](include/zipper/transform/README.md)
for the full API reference.

## Utilities

Zipper provides matrix decompositions, iterative solvers, and Krylov
subspace methods in the `zipper::utils` namespace.

### Matrix Decompositions

```cpp
#include <zipper/utils/decomposition/qr.hpp>
#include <zipper/utils/decomposition/lu.hpp>
#include <zipper/utils/decomposition/llt.hpp>
#include <zipper/utils/decomposition/ldlt.hpp>

zipper::Matrix<double, 3, 3> A{{4,-1,0},{-1,4,-1},{0,-1,4}};
zipper::Vector<double, 3> b{1, 2, 3};

auto [Q, R] = zipper::utils::qr(A);          // Householder QR
auto plu = zipper::utils::plu(A);             // PLU (partial pivoting)
auto x1 = plu.solve(b);                       // solve via PLU

auto L = zipper::utils::llt(A);               // Cholesky (SPD matrices)
auto x2 = zipper::utils::llt_solve(A, b);

auto [Ld, D] = zipper::utils::ldlt(A);        // LDLT (avoids square roots)
auto x3 = zipper::utils::ldlt_solve(A, b);
```

### Iterative Solvers

All iterative solvers return `std::expected<SolverResult, SolverError>`.

```cpp
#include <zipper/utils/solver/conjugate_gradient.hpp>
#include <zipper/utils/solver/gmres.hpp>
#include <zipper/utils/solver/bicgstab.hpp>

// Conjugate Gradient (SPD matrices)
auto result = zipper::utils::solver::conjugate_gradient(A, b, 1e-10, 100);

// GMRES (general non-symmetric)
auto result2 = zipper::utils::solver::gmres(A, b, 1e-10, 100);

// BiCGSTAB (general non-symmetric, no transpose needed)
auto result3 = zipper::utils::solver::bicgstab(A, b, 1e-10, 100);
```

Additional solvers: Preconditioned CG (`pcg`), Jacobi, Gauss-Seidel,
Multigrid. Preconditioners: `JacobiPreconditioner`, `SSORPreconditioner`.

### Krylov Methods

```cpp
#include <zipper/utils/krylov/power_method.hpp>
#include <zipper/utils/krylov/arnoldi.hpp>
#include <zipper/utils/krylov/lanczos.hpp>

// Dominant eigenvalue/eigenvector via power iteration
auto [eigenvalue, eigenvector] = zipper::utils::power_method(A);

// Arnoldi iteration (general matrices)
auto [Q, H] = zipper::utils::arnoldi(A, v0, k);

// Lanczos iteration (symmetric matrices)
auto [Q, T] = zipper::utils::lanczos(A, v0, k);
```

### SuiteSparse Bindings (Optional)

When built with `use_suitesparse=true`, Zipper provides bindings for
CHOLMOD, UMFPACK, and SPQR for large sparse systems. Input matrices must be
in CSC format.

```cpp
#include <zipper/utils/suitesparse/cholmod.hpp>
#include <zipper/utils/suitesparse/umfpack.hpp>
#include <zipper/utils/suitesparse/spqr.hpp>

auto x = zipper::utils::cholmod_solve(A_csc, b);   // sparse Cholesky (SPD)
auto x = zipper::utils::umfpack_solve(A_csc, b);   // sparse LU (general)
auto x = zipper::utils::spqr_solve(A_csc, b);      // sparse QR (least-squares)
```

See [`include/zipper/utils/README.md`](include/zipper/utils/README.md) for
the full API reference.

## Dependencies

Zipper depends on [mdspan](https://github.com/kokkos/mdspan). This
dependency should disappear as C++26 `std::mdspan` becomes available in
standard libraries.

For building and testing, Zipper depends on [Meson](https://mesonbuild.com)
and [Catch2](https://catch2.org), and is ready for use with [Conan](https://conan.io).

## Testing

If you have your own installations of all of the above dependencies you can
build by

```bash
meson setup .. . -Dtesting=true
meson test -C build # build and test
```

```bash
# prepare conan
conan install . --output-folder=build/conan --build=missing 
# enter build directory
pushd build 
# configure meson to use the output of conan
meson setup --native-file conan/conan_meson_native.ini .. . -Dtesting=true 
ninja # build
```

## Type Hierarchy

```
Expression nodes (lazy evaluation tree)
  NullaryExpression    MDArray, MDSpan, Constant, StaticConstant, Identity, Unit, Random
  UnaryExpression      Slice, Swizzle, Reshape, Cast, Abs, DiagonalEmbed, ExtentView, UnsafeRef, ...
  BinaryExpression     Addition, MatrixProduct, TensorProduct, HamiltonProduct, ...
  Reduction            Sum, Trace, Determinant, Norm, Contraction, ...

User-facing wrappers (CRTP, semantic layer)
  DataArrayBase        plain storage, no arithmetic
    TensorBase         generic rank-N tensors
      ArrayBase        coefficient-wise operations (+, *, abs, ...)
      MatrixBase       matrix algebra (product, inverse, decomposition)
      VectorBase       column vectors (dot, cross, norm)
      FormBase         row vectors / covectors (contraction with vectors)
    QuaternionBase     quaternion algebra (Hamilton product, conjugate, slerp)
```

**Concrete owning types** (static extents):
`DataArray<T, N...>`, `Tensor<T, N...>`, `Array<T, N...>`,
`Vector<T, N>`, `Matrix<T, R, C>`, `Form<T, N>`, `Quaternion<T>`.

**Sparse types:**
`COOMatrix<T, R, C>`, `COOVector<T, N>`, `CSMatrix<T, R, C>`,
`CSRMatrix<T, R, C>`, `CSVector<T, N>`.

## Comparison with Eigen and Blaze

Zipper's expression template architecture addresses several lifetime-safety
problems that arise in Eigen (and, to a lesser extent, Blaze).

### Dangling references from temporaries

In Eigen, capturing an expression that references a temporary is silently
unsafe:

```cpp
// Eigen -- UNDEFINED BEHAVIOUR (temporary dies at end of statement)
Eigen::Vector3d a(1, 2, 3);
auto expr = a + Eigen::Vector3d(4, 5, 6);  // temporary is destroyed
std::cout << expr;                          // reads dead memory
```

In Zipper, rvalue operands are automatically moved into the expression tree:

```cpp
// Zipper -- safe (temporary is owned by the expression)
zipper::Vector<double, 3> a{1, 2, 3};
auto expr = a + zipper::Vector<double, 3>{4, 5, 6};
// expr(0) == 5.0 -- the temporary lives inside expr
```

### Views that escape their scope

Eigen's `.col()` / `.row()` / `.block()` return lightweight views that hold
a raw pointer. Nothing prevents you from returning them:

```cpp
// Eigen -- compiles, but the view dangles if the matrix goes out of scope
auto get_col(Eigen::Matrix3d& M) { return M.col(0); }
```

In Zipper, views inherit `NonReturnable` (deleted copy constructor) so this
is a compile error. You must explicitly opt in with `.unsafe()`:

```cpp
// Zipper -- compile error: copy constructor deleted
auto get_col(zipper::Matrix<double, 3, 3>& M) {
    auto c = M.col(zipper::index_type(0));
    return c;  // ERROR
}

// Explicit opt-in: caller asserts M outlives the view
auto get_col_unsafe(zipper::Matrix<double, 3, 3>& M) {
    return M.col(zipper::index_type(0)).unsafe();  // OK
}
```

### Summary of safety model

| Scenario | Eigen | Zipper |
|---|---|---|
| `auto e = a + temp()` | UB (dangling) | Safe (temp moved in) |
| `auto v = M.col(j); auto v2 = v;` | Compiles (shallow copy) | Compile error (copy deleted) |
| Return a view from a function | Compiles (dangling) | Compile error; use `.unsafe()` |
| Materialise to owned data | `.eval()` | `.eval()` |
| Deep-copy lazy tree | N/A | `.to_owned()` |
