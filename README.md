
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
  NullaryExpression    MDArray, MDSpan, Constant, Identity, Unit, Random
  UnaryExpression      Slice, Swizzle, Reshape, Cast, Abs, UnsafeRef, ...
  BinaryExpression     Addition, MatrixProduct, TensorProduct, ...
  Reduction            Sum, Trace, Determinant, Norm, Contraction, ...

User-facing wrappers (CRTP, semantic layer)
  DataArrayBase        plain storage, no arithmetic
    TensorBase         generic rank-N tensors
      ArrayBase        coefficient-wise operations (+, *, abs, ...)
      MatrixBase       matrix algebra (product, inverse, decomposition)
      VectorBase       column vectors (dot, cross, norm)
      FormBase         row vectors / covectors (contraction with vectors)
```

**Concrete owning types** (static extents):
`DataArray<T, N...>`, `Tensor<T, N...>`, `Array<T, N...>`,
`Vector<T, N>`, `Matrix<T, R, C>`, `Form<T, N>`.

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
