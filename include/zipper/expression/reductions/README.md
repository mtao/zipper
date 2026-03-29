# Reductions

Reductions collapse an expression to a lower-rank result (often a scalar).
They all inherit from `ReductionBase<Derived, Child>`.

---

## Scalar Reductions

These reduce an entire expression to a single scalar value.

| Header | Expression | Formula | Description |
|--------|-----------|---------|-------------|
| `CoefficientSum.hpp` | `CoefficientSum<Expr>` | sum of all coefficients | Used by `.sum()` |
| `CoefficientProduct.hpp` | `CoefficientProduct<Expr>` | product of all coefficients | Used by `.prod()` |
| `Trace.hpp` | `Trace<Expr>` | sum of diagonal elements | Matrix trace; requires rank-2 square input |
| `Determinant.hpp` | `Determinant<Expr>` | det(A) | Matrix determinant via cofactor expansion |
| `LpNorm.hpp` | `LpNorm<Expr, P>` | (sum |x_i|^P)^(1/P) | Lp norm; specialised for P=1 (sum of abs) and P=2 (Euclidean) |
| `LpNormPowered.hpp` | `LpNormPowered<Expr, P>` | sum |x_i|^P | Lp norm without the final root (avoids the pow call) |
| `Contraction.hpp` | `Contraction<Expr>` | full tensor contraction | Fold-in-half contraction: pairs index 0 with N/2, 1 with N/2+1, etc. |

## Boolean Reductions

| Header | Expression | Description |
|--------|-----------|-------------|
| `All.hpp` | `All<Expr>` | True if all coefficients are truthy |
| `Any.hpp` | `Any<Expr>` | True if any coefficient is truthy |

---

## Tensor Contraction

`Contraction<Expr>` implements full tensor contraction for even-rank tensors.
It splits the indices in half and sums over matching pairs:

```
result = sum_{i,j,...} expr(i, j, ..., i, j, ...)
```

For a rank-2 tensor this is the trace. For a rank-4 tensor with extents
(a,b,a,b), it contracts indices (0,2) and (1,3) simultaneously.

Free functions (in `TensorBase.hxx`):
- `full_contract(tensor)` -- wraps in `Contraction`
- `contract<I, J>(tensor)` -- wraps in `PartialTrace` (see `unary/PartialTrace.hpp`)
- `tensor_product(a, b)` -- wraps in `TensorProduct` (see `binary/TensorProduct.hpp`)

---

## Support Files

| Path | Purpose |
|------|---------|
| `ReductionBase.hpp` | CRTP base class for all reductions |
| `detail/swap_parity.hpp` | Permutation sign computation (used by Determinant) |
