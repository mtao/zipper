# Unary Expressions

Unary expressions take a single child expression and produce a new expression.
They all inherit from `UnaryExpressionBase<Derived, ChildExpr>` and store the
child via `expression_storage_t<ChildExpr>` (lvalue ref by reference, rvalue by
value).

This directory contains 19 expression types plus the CRTP base, organized below
by what they do to the index space.

---

## Index-Space Manipulation

These expressions change *which* child indices are accessed, without
transforming coefficient values.

### Single-Axis Remapping (IndexSet participants)

IndexSets describe which indices are "active" along a single dimension. These
expressions propagate or define IndexSet information, enabling sparse-aware
optimizations (e.g., skipping known-zero entries in `MatrixProduct`).

| Header | Expression | What it does | Rank change | IndexSet |
|--------|-----------|--------------|-------------|----------|
| `Slice.hpp` | `Slice<Expr, Slices...>` | Sub-view via per-dimension slices (strided, fixed, full, array, vector, expression) | Rank - (# integer slices) | Propagated from child via `_remap_range` |
| `TriangularView.hpp` | `TriangularView<Mode, Expr>` | Triangular mask on rank-2 matrix; zeros outside the triangle | Same (rank 2) | Own implementation (returns row-dependent ranges) |
| `UnsafeRef.hpp` | `UnsafeRef<Child>` | Transparent identity wrapper that overrides `stores_references` | Same | Propagated from child |

**Slice** is the most complex expression in the library (~760 lines). It
supports six slice types per dimension: `strided_slice`, `full_extent_t`,
integer (fixed index, removes the dimension), `std::array`, `Vector`, and
arbitrary expressions. The `_remap_range` method translates child IndexSets
through the slice mapping using `range_intersection()` and
`to_index_set()`.

**TriangularView** is the canonical example of an expression that *introduces*
sparsity: for `Upper` mode, column indices below the diagonal are reported as
zero without querying the child.

### Multi-Dimension Permutation

| Header | Expression | What it does | Rank change | IndexSet |
|--------|-----------|--------------|-------------|----------|
| `Swizzle.hpp` | `Swizzle<Expr, Indices...>` | Permutes/reorders dimensions; `coeff(i,j,k)` -> `child.coeff(I(i),J(j),K(k))` | = `sizeof...(Indices)` | Propagated (reorders child's per-dimension IndexSets) |

Swizzle operates on the full dimension tuple rather than individual axes.
It reindexes the child's IndexSets to match the permuted dimension order.

### Dimension Insertion / Removal

These expressions change the rank by adding or collapsing dimensions.

| Header | Expression | What it does | Rank change | IndexSet |
|--------|-----------|--------------|-------------|----------|
| `Repeat.hpp` | `Repeat<Mode, Count, Child>` | Tiles/broadcasts by prepending (`Left`) or appending (`Right`) dimensions | Rank + `Count` | Propagated: child dims from child, repeated dims return `FullRange` |
| `AntiSlice.hpp` | `AntiSlice<Expr, Dims...>` | Inserts new size-1 dimensions (inverse of integer-slice in `Slice`) | Rank + `sizeof...(Dims)` | No |
| `Diagonal.hpp` | `Diagonal<Expr>` | Extracts diagonal: `coeff(i)` -> `child.coeff(i, i, ...)` | Always rank 1 | No |
| `Homogeneous.hpp` | `Homogeneous<Mode, Child>` | Extends dimension 0 by 1 for homogeneous coordinates (`Append` or `Prepend`) | Same rank, dim 0 grows by 1 | No |
| `Reshape.hpp` | `Reshape<Expr, NewExtents>` | Reinterprets shape via linearize + unravel (row-major) | Arbitrary | No |

**Repeat** propagates IndexSets from the child for child dimensions. Repeated
(broadcast) dimensions are fully dense — all indices are active because the
value is the same regardless of the index along that dimension. When the child
is rank-1 (e.g., a Unit vector broadcast to rank-2), the child's no-arg
`index_set<0>()` is called since the sparsity pattern is independent of the
broadcast index.

### Dimension Reduction

These collapse one or more dimensions by aggregating values.

| Header | Expression | What it does | Rank change |
|--------|-----------|--------------|-------------|
| `PartialReduction.hpp` | `PartialReduction<Expr, Red, Indices...>` | Collapses specified dimensions via a reduction functor (sum, max, etc.) | Rank - `sizeof...(Indices)` |
| `PartialTrace.hpp` | `PartialTrace<Expr, Indices...>` | Contracts dimension pairs (generalized trace) | Rank - `sizeof...(Indices)` |

---

## Value Transformation

These expressions apply a function to each coefficient without changing the
index space. The output has the same extents as the child.

### Generic Coefficient-Wise Operations

| Header | Expression | What it does | IndexSet |
|--------|-----------|--------------|----------|
| `CoefficientWiseOperation.hpp` | `CoefficientWiseOperation<Child, Op>` | Applies `Op` to each coefficient | Conditional: propagated if `Op` is zero-preserving |
| `ScalarOperation.hpp` | `ScalarOperation<Child, Op, Scalar, Side>` | Applies `Op(coeff, scalar)` or `Op(scalar, coeff)` | Conditional: propagated if `Op` is zero-preserving |

**Zero-preserving operations** are those where `Op(0) == 0` (unary) or
`Op(0, scalar) == 0` (binary). The library tracks this via
`is_zero_preserving_operation_v<Op>` and
`is_zero_preserving_scalar_operation_v<Op, Side>` traits. When an operation is
zero-preserving, the child's IndexSet is propagated through, enabling
sparse-aware evaluation. Otherwise, all indices are assumed potentially nonzero.

### Concrete Aliases and Specializations

These are all thin wrappers around `CoefficientWiseOperation` or
`ScalarOperation`:

| Header | Expression | Base | Zero-preserving? |
|--------|-----------|------|------------------|
| `ScalarArithmetic.hpp` | `Negate<E>` | `CoefficientWiseOperation<E, std::negate<>>` | Yes |
| `ScalarArithmetic.hpp` | `ScalarMultiplies<E, S>` | `ScalarOperation<E, std::multiplies<>, S, Right>` | Yes |
| `ScalarArithmetic.hpp` | `ScalarDivides<E, S>` | `ScalarOperation<E, std::divides<>, S, Right>` | Yes (scalar on right only) |
| `ScalarArithmetic.hpp` | `ScalarPlus<E, S>` | `ScalarOperation<E, std::plus<>, S, Side>` | No |
| `ScalarArithmetic.hpp` | `ScalarMinus<E, S>` | `ScalarOperation<E, std::minus<>, S, Side>` | No |
| `Abs.hpp` | `Abs<E>` | `CoefficientWiseOperation<E, detail::Abs>` | Could be (not yet registered) |
| `Cast.hpp` | `Cast<A, E>` | `CoefficientWiseOperation<E, ...>` | Could be (not yet registered) |
| `ScalarPower.hpp` | `ScalarPower<E, S>` | `ScalarOperation<E, detail::ScalarPower, S, Right>` | No (`pow(0,p)` not always 0) |
| `Identity.hpp` | `Identity<E>` (unary) | `CoefficientWiseOperation<E, std::identity>` | Could be (not yet registered) |

### Structural Value Computation

| Header | Expression | What it does | IndexSet |
|--------|-----------|--------------|----------|
| `Cofactor.hpp` | `Cofactor<Expr>` | Computes the cofactor matrix (signed minor determinants) | No |

---

## Support Files

| Path | Purpose |
|------|---------|
| `UnaryExpressionBase.hpp` | CRTP base class; stores child via `expression_storage_t`, provides `child()` accessor |
| `concepts/ScalarOperation.hpp` | Concept for scalar operation expressions |
| `detail/CoeffWiseTraits.hpp` | Traits for deducing `value_type` of coefficient-wise operations |
| `detail/invert_integer_sequence.hpp` | Compile-time integer sequence inversion (used by `Swizzle`, `AntiSlice`) |
| `detail/operation_implementations.hpp` | Functor structs: `Abs`, `ScalarPower`, min/max (some use `static operator()`) |
| `detail/PartialReductionDispatcher.hpp` | Dispatch logic for `PartialReduction` over static/dynamic dimension sets |

---

## IndexSet Integration Summary

The **IndexSet** system (defined in `expression/detail/IndexSet.hpp`) describes
which indices along a single axis may have nonzero values. Expressions opt in
by setting `has_index_set = true` in their `ExpressionTraits` and implementing
`index_set<D>()`.

Propagation through unary expressions:

```
Child IndexSet
     |
     v
  Slice          -- intersects child IndexSet with slice range, remaps to output space
  TriangularView -- generates own IndexSet (row-dependent column ranges)
  Swizzle        -- permutes child's per-dimension IndexSets
  Repeat         -- child dims from child, repeated dims return FullRange
  UnsafeRef      -- passes through unchanged
  CWiseOp/ScalarOp -- passes through IF operation is zero-preserving
```

Expressions that do NOT propagate IndexSets (AntiSlice, Diagonal, Homogeneous,
Reshape, PartialReduction, PartialTrace, Cofactor) could potentially be
extended in the future but currently report all indices as potentially nonzero.
