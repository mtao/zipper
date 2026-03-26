# Nullary Expressions

Nullary expressions are leaf nodes in the expression tree -- they produce
values without any child expressions. They all inherit from
`NullaryExpressionBase<Derived>`.

---

## Data-Backed Expressions

These hold actual data in memory and provide `coeff_ref()` for mutable access.

| Header | Expression | Storage | Description |
|--------|-----------|---------|-------------|
| `MDArray.hpp` | `MDArray<T, Extents, Layout, Accessor>` | Owning (`DenseData`) | The primary owning dense array; backs `Vector`, `Matrix`, `DataArray`, etc. |
| `MDSpan.hpp` | `MDSpan<T, Extents, Layout, Accessor>` | Non-owning (`SpanData`) | Non-owning view into contiguous memory; backs `as_span()` and span-based construction |
| `StlMDArray.hpp` | `StlMDArray<Container>` | STL container | Wraps `std::vector`, `std::array`, etc.; used by initialiser-list constructors |
| `LinearLayoutExpression.hpp` | (base class) | (template parameter) | Common base for MDArray and MDSpan; provides linear indexing, `data()`, `as_std_span()` |

`MDArray` and `MDSpan` both inherit from `LinearLayoutExpression`, which maps
multidimensional indices to a flat offset via a layout policy (left-major or
right-major). `MDArray` owns its storage via `DenseData` (static `std::array`
or dynamic `std::vector`), while `MDSpan` holds a non-owning `SpanData`.

---

## Computed Expressions

These compute their coefficients on-the-fly from parameters (no stored data).

| Header | Expression | Description |
|--------|-----------|-------------|
| `Constant.hpp` | `Constant<T, Indices...>` | Returns the same value for all indices |
| `Identity.hpp` | `Identity<T, N>` | Kronecker delta: 1 on diagonal, 0 elsewhere |
| `Iota.hpp` | `Iota<T, D, Indices...>` | Returns the D-th index cast to T (counting sequence along an axis) |
| `Unit.hpp` | `Unit<T, Extent, IndexType>` | Unit (basis) vector: 1 at one index, 0 elsewhere |
| `Random.hpp` | `Random<T, Indices...>` | Random values (generated once and cached) |
| `StaticConstant.hpp` | `StaticConstant<T, Value, Indices...>` | Returns `static_cast<T>(Value)` for all indices, where `Value` is a compile-time `constexpr int`; aliases: `Zero<T, Indices...>` (Value=0), `Ones<T, Indices...>` (Value=1). When Value==0, has `has_index_set = true` with empty index sets for zero-aware optimisation. Unlike `Constant`, the value is encoded in the type. |

### Factory Functions

- `unit_vector<T, size, index>()` -- static extent + static index
- `unit_vector<T, size>(index)` -- static extent + dynamic index
- `unit_vector<T>(size, index)` -- dynamic extent + dynamic index
- `iota<T, D>(start, extents)` -- counting sequence starting at `start` along axis D (Iota + Constant via binary::Plus)
- `linear_space<T, D>(start, stop, extents)` -- evenly spaced values from `start` to `stop` (inclusive) along axis D

### Zero-Aware Sparsity

`Unit` and `Identity` have `has_index_set = true` in their ExpressionTraits,
providing `index_set<D>()` methods that describe which indices are structurally
nonzero. This enables `MatrixVectorProduct` to skip known-zero entries when
multiplying by a unit vector or identity matrix.

---

## Support Files

| Path | Purpose |
|------|---------|
| `NullaryExpressionBase.hpp` | CRTP base class for all nullary expressions |
