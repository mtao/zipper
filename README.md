
# Zipper

A zippy C++26 tensor processing library. Heavily inspired from [Eigen](https://eigen.tuxfamily.org/) and [mdpsan](https://cppreference.com/utility/mdspan), this library-in-development intends to provide a light-weight clean syntax for linear algebra and tensor algebra while also providing convenient backends to transform between different data representations.

## Principles

The underlying concept in Zipper is a _view_, which merely represents an underlying map $$\mathbb{Z}^r \rightarrow T$$ for some rank $$r$$. When a view represents a (multi-dimensional) array of objects. The semantics of vectors (r=1), matrices (r=2), tensors, or forms are induced by wrapping that view in a container for which those semantics are induced.

More concretely, we can take unit vectors as

```cpp
#include <zipper/views/nullary/UnitView.hpp>

// vector {0,1,0}
auto e1 = zipper::views::nullary::unit_view<double,3>(1);
```

This view can be used by multiple types of semantics, like the semantics of a vector or form. Note that below deduction guides are used to maintain proper ownership of the view (e1 is a non-const ref so the `*Base` classes do not own any data, but instead hold a reference to `e1` internally).

```cpp
#include <zipper/VectorBase.hpp>
#include <zipper/FormBase.hpp>

// a vector whose data is the view e1
zipper::VectorBase x = e1;

// a 1-form, equivalently a row-vector whose data is the view e1
zipper::FormBase u = e1;
```

With these semantic wrappers we can now do semantic-specific operations

```cpp
// form * vector is like a dot product, resulting in a scalar
double res = u * x;

// a vector holding a view that holds the expression for (2 * x)
zipper::VectorBase x2 = 2 * x;

// a vector that owns its own data
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

##### `unsafe()` -- Returnable Views (Caller Asserts Lifetime)

`unsafe()` wraps the expression in an `UnsafeRef` node that overrides
`stores_references` to `false`, making the result copyable and returnable. The
caller asserts that the referenced data will outlive all uses. This adds minor
verbosity, but the trade-off is intentional: rather than silently allowing
dangling references, Zipper requires explicit opt-in for potentially unsafe
lifetime escapes.

```cpp
zipper::Matrix<double, 3, 3> M{{1,2,3},{4,5,6},{7,8,9}};

// With unsafe(): the view becomes returnable
auto s = M.col(zipper::index_type(1)).unsafe();
s(0) = 42.0;                    // writes through to M(0,1)
M(2, 1) = 99.0;                 // visible through s(2)
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
| `unsafe()` | Returnable view wrapper | References original data | You need `auto s = M.col(j)` ergonomics |

## Dependencies

Zipper depends on [fmt](https://fmt.dev) and [mdspan](https:://github.com/kokkos/mdspan), but both of these dependencies should disappear as c++26 functionality becomes more common in existing libraries.

For buliding and testing zipper currently depends on [meson](https://mesonbuild.com)
and [Catch2](https://catch2.org), and is ready for use with [conan](https://conan.io).

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
