# Binary Expressions

Binary expressions combine two child expressions to produce a new expression.
They all inherit from `BinaryExpressionBase<Derived, LHS, RHS>` and store each
child via `expression_storage_t` (lvalue by reference, rvalue by value).

---

## Coefficient-Wise Arithmetic

| Header | Expression | Description |
|--------|-----------|-------------|
| `Operation.hpp` | `Operation<LHS, RHS, Op>` | Generic coefficient-wise binary operation |
| `ZeroAwareOperation.hpp` | `ZeroAwareOperation<LHS, RHS, Op>` | Sparse-aware variant that skips known-zero regions |
| `ArithmeticExpressions.hpp` | `Addition`, `Subtraction`, `CoefficientWiseProduct`, `CoefficientWiseDivision` | Standard arithmetic; Addition and Subtraction use `ZeroAwareOperation` for IndexSet propagation |

The arithmetic expressions are created by the `operator+`, `operator-`,
`operator*`, and `operator/` overloads defined in the `*Base.hxx` files.
`CoefficientWiseProduct` and `CoefficientWiseDivision` are element-wise
(Hadamard) operations available through `ArrayBase`, not matrix multiplication.

---

## Algebraic Products

These implement products with specific algebraic semantics that change the
output shape.

| Header | Expression | Input Ranks | Output Rank | Description |
|--------|-----------|-------------|-------------|-------------|
| `MatrixProduct.hpp` | `MatrixProduct<A, B>` | (M,K) x (K,N) | (M,N) | Standard matrix-matrix product with IndexSet-aware inner loop |
| `MatrixVectorProduct.hpp` | `MatrixVectorProduct<Mat, Vec>` | (M,K) x (K,) | (M,) | Matrix-vector product; uses IndexSet for sparse-aware dot products |
| `TensorProduct.hpp` | `TensorProduct<A, B>` | (a...) x (b...) | (a..., b...) | Outer (tensor/Kronecker) product; concatenates index spaces |
| `FormTensorProduct.hpp` | `FormTensorProduct<A, B>` | (a...) x (b...) | scalar or contracted | Tensor product composed with partial trace; used by Form * Vector (dot product) |
| `CrossProduct.hpp` | `CrossProduct<A, B>` | (3,) x (3,) | (3,) | 3D vector cross product |
| `WedgeProduct.hpp` | `WedgeProduct<A, B>` | (N,) x (N,) | (N,N) | Exterior (wedge) product; antisymmetric rank-2 result |

### Tensor Product and Contraction

`TensorProduct<A, B>` concatenates the index spaces of two tensors:
`A(i,j) * B(k,l)` becomes `TP(i,j,k,l) = A(i,j) * B(k,l)`.

`FormTensorProduct<A, B>` composes `TensorProduct` with `PartialTrace` to
contract matching indices. This is the mechanism behind `Form * Vector`
producing a scalar.

Free functions for ergonomic use (defined in `TensorBase.hxx`):
- `tensor_product(a, b)` -- explicit outer product
- `contract<I, J>(tensor)` -- trace over index pair I, J
- `full_contract(tensor)` -- fold-in-half contraction to scalar

---

## Support Files

| Path | Purpose |
|------|---------|
| `BinaryExpressionBase.hpp` | CRTP base; stores LHS and RHS children, provides `lhs()` / `rhs()` |
| `detail/CoeffWiseTraits.hpp` | Traits for deducing value_type of binary operations |
| `detail/operation_implementations.hpp` | Functor structs used by Operations |
| `detail/minmax.hpp` | Min/max functors |
| `detail/pivot_utils.hpp` | Pivot utilities for decomposition algorithms |
