# zipper::expression

The expression layer is the core of Zipper's lazy evaluation engine. Every
computation is represented as a tree of expression nodes that are evaluated
on-demand when coefficients are accessed.

## Architecture

```
ExpressionBase<Derived>                        (CRTP root)
  ├── NullaryExpressionBase<D>                 (leaf nodes)
  │     MDArray, MDSpan, Constant, Identity, Unit, Random
  ├── UnaryExpressionBase<D, Child>            (single-child transforms)
  │     Slice, Swizzle, Reshape, Cast, Abs, UnsafeRef, ...
  ├── BinaryExpressionBase<D, LHS, RHS>        (two-child operations)
  │     Addition, MatrixProduct, TensorProduct, ...
  └── ReductionBase<D, Child>                  (whole-expression reductions)
        Sum, Trace, Determinant, Norm, Contraction, ...
```

## Expression Interface

Every expression provides:

- `rank()` -- number of dimensions (compile-time constant)
- `extent(rank_type d)` -- size along dimension `d`
- `extents()` -- the full `std::extents<...>` object
- `coeff(index_type... i) const -> value_type` -- coefficient by value

Writable expressions (`WritableExpression` concept) additionally provide:

- `coeff_ref(index_type... i) -> value_type&` -- mutable reference
- `const_coeff_ref(index_type... i) const -> value_type const&`
- `assign(const Other&)` -- bulk assignment from another expression

## Value Category and Ownership

Child expressions are stored via `expression_storage_t<T>`:

- **Lvalue references** are stored by reference (cheap, zero-copy)
- **Rvalue temporaries** are moved in and stored by value (owned)

This means expression trees automatically own their temporaries, preventing
the dangling-reference bugs common in Eigen.

Expressions that hold references have `stores_references = true` in their
`ExpressionTraits`. At the wrapper level (`ZipperBase`), this causes the
type to inherit `NonReturnable` (deleted copy constructor), preventing
accidental escapes from scope.

## ExpressionTraits

Each expression type specialises `ExpressionTraits<T>`, which provides:

- `value_type` -- scalar type
- `extents_type` -- `std::extents<...>` type
- `is_writable` -- can coefficients be mutated?
- `stores_references` -- does the expression hold references?
- `has_index_set` -- does it provide sparse index information?
- `is_resizable` -- can the extents change at runtime?

## Subdirectories

| Directory | Contents | README |
|-----------|----------|--------|
| `nullary/` | Leaf nodes (data sources) | [nullary/README.md](nullary/README.md) |
| `unary/` | Single-child transforms | [unary/README.md](unary/README.md) |
| `binary/` | Two-child operations | [binary/README.md](binary/README.md) |
| `reductions/` | Whole-expression reductions | [reductions/README.md](reductions/README.md) |
| `concepts/` | Expression capability concepts (`WritableExpression`, etc.) | -- |
| `detail/` | Traits, assignment helpers, index sets | -- |
