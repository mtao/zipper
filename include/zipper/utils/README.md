# Utilities

Algorithms, decompositions, iterative solvers, and helper functions that
operate on Zipper types. Everything lives in the `zipper::utils` namespace
(or sub-namespaces thereof).

---

## Top-Level Utilities

| Header | Function(s) | Description |
|--------|------------|-------------|
| `determinant.hpp` | `determinant(M)` | Matrix determinant (delegates to `expression::reductions::Determinant`) |
| `inverse.hpp` | `inverse(M)` | Matrix inverse; closed-form for 2x2/3x3, QR-based for general nxn |
| `format.hpp` | `zipper_format_string(M)` | `std::format` support for all Zipper types (rank-0 scalar, rank-1 vector, rank-2+ matrix/tensor) |
| `max_coeff.hpp` | `maxCoeff(d)`, `maxCoeffIndex(d)` | Maximum coefficient value and its multi-index |
| `min_coeff.hpp` | `minCoeff(d)`, `minCoeffIndex(d)` | Minimum coefficient value and its multi-index |
| `mean_coeff.hpp` | `meanCoeff(d)` | Arithmetic mean of all coefficients |

---

## Decompositions (`decomposition/`)

Direct matrix factorisations. Each returns a result struct containing the
factors, and most provide a `solve(b)` method for solving linear systems.

| Header | Function(s) | Description |
|--------|------------|-------------|
| `decomposition/qr.hpp` | `qr(A)`, `qr_full(A)`, `qr_gram_schmidt(A)` | QR decomposition: reduced Householder, full Householder, and classical Gram-Schmidt variants. Returns Q and R. |
| `decomposition/lu.hpp` | `plu(A)`, `plu_solve(A, b)` | PLU decomposition (LU with partial pivoting). `PLUResult::solve(b)` for forward/back substitution. |
| `decomposition/llt.hpp` | `llt(A)`, `llt_solve(A, b)` | Cholesky (LLT) factorisation for SPD matrices. Returns lower triangular factor L. |
| `decomposition/ldlt.hpp` | `ldlt(A)`, `ldlt_solve(A, b)` | LDLT factorisation for symmetric positive semi-definite matrices. Avoids square roots (unlike LLT). |

---

## Iterative Solvers (`solver/`)

Iterative methods for solving linear systems `Ax = b`. All solvers return
`std::expected<SolverResult<T, Dim>, SolverError>`.

| Header | Description |
|--------|-------------|
| `solver/result.hpp` | `SolverResult<T, Dim>` and `SolverError` -- common result/error types for all iterative solvers |

### Krylov Solvers

| Header | Solver | Matrix Requirements | Description |
|--------|--------|-------------------|-------------|
| `solver/conjugate_gradient.hpp` | Conjugate Gradient (CG) | SPD | Gold-standard for SPD systems; 1 matvec per iteration, guaranteed convergence in n iterations |
| `solver/pcg.hpp` | Preconditioned CG (PCG) | SPD | CG with a preconditioner to cluster eigenvalues and accelerate convergence |
| `solver/gmres.hpp` | GMRES | General | Generalised Minimum Residual via Arnoldi + Givens rotations; works for non-symmetric systems |
| `solver/bicgstab.hpp` | BiCGSTAB | General | Bi-Conjugate Gradient Stabilised; 2 matvecs per iteration, no transpose needed |

### Stationary Solvers

| Header | Solver | Matrix Requirements | Description |
|--------|--------|-------------------|-------------|
| `solver/jacobi.hpp` | Jacobi | Diagonally dominant | Simple stationary method; inherently parallel |
| `solver/gauss_seidel.hpp` | Gauss-Seidel | Diag. dominant or SPD | Improves on Jacobi by using updated values immediately |
| `solver/multigrid.hpp` | Geometric Multigrid | SPD (1D hierarchy) | V-cycle with damped Jacobi smoothing and PLU at coarsest level; `MultigridLevel<T>`, `vcycle()`, `multigrid()`, `build_1d_hierarchy()` |

### Preconditioners (`solver/preconditioner/`)

| Header | Preconditioner | Description |
|--------|---------------|-------------|
| `solver/preconditioner/jacobi_preconditioner.hpp` | `JacobiPreconditioner` | Diagonal preconditioner: `z_i = r_i / A_{ii}`; stores inverse diagonal |
| `solver/preconditioner/ssor_preconditioner.hpp` | `SSORPreconditioner` | Symmetric Successive Over-Relaxation with configurable omega |

### Support Files

| Path | Purpose |
|------|---------|
| `solver/detail/triangular_substitute.hpp` | Forward and back substitution for triangular systems |

---

## Krylov Methods (`krylov/`)

Building blocks for Krylov subspace methods. These are used internally by
GMRES and can be used directly for eigenvalue estimation.

| Header | Function | Description |
|--------|----------|-------------|
| `krylov/arnoldi.hpp` | `arnoldi(A, v, k)` | Arnoldi iteration for general matrices; builds orthonormal basis Q and upper Hessenberg H |
| `krylov/lanczos.hpp` | `lanczos(A, v, k)` | Lanczos iteration for symmetric matrices; produces symmetric tridiagonal T with O(n) per step |
| `krylov/power_method.hpp` | `power_method(A, ...)` | Power iteration for the dominant eigenvalue/eigenvector; returns Rayleigh quotient and eigenvector |

---

## Orthogonalisation (`orthogonalization/`)

| Header | Function | Description |
|--------|----------|-------------|
| `orthogonalization/gram_schmidt.hpp` | `gram_schmidt(M)`, `gram_schmidt_in_place(M)` | Classical Gram-Schmidt orthogonalisation; zeroes degenerate columns |

---

## SuiteSparse Bindings (`suitesparse/`)

Optional bindings for [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html)
direct solvers. Available only when `ZIPPER_HAS_SUITESPARSE` is defined
(enabled via the `use_suitesparse` Meson option). Input matrices must be in
CSC format.

| Header | Function(s) | Description |
|--------|------------|-------------|
| `suitesparse/common.hpp` | `CholmodCommon`, `to_cholmod_sparse()`, `to_cholmod_dense()`, `from_cholmod_dense()` | RAII wrappers and conversion helpers for SuiteSparse types |
| `suitesparse/cholmod.hpp` | `cholmod_factor(A)`, `cholmod_solve(A, b)` | CHOLMOD sparse Cholesky for SPD matrices |
| `suitesparse/umfpack.hpp` | `umfpack_factor(A)`, `umfpack_solve(A, b)` | UMFPACK sparse LU for general square matrices |
| `suitesparse/spqr.hpp` | `spqr_factor(A)`, `spqr_solve(A, b)`, `spqr_qr(A)` | SuiteSparseQR sparse QR for general (possibly rectangular) matrices; supports least-squares |

---

## Extents Utilities (`extents/`)

Compile-time and runtime utilities for working with `std::extents<...>`.

| Header | Function / Type | Description |
|--------|----------------|-------------|
| `extents/all_extents_indices.hpp` | `all_extents_indices(e)` | Cartesian product of all valid index tuples for given extents |
| `extents/as_array.hpp` | `as_array(e)` | Convert extents to `std::array<index_type, N>` |
| `extents/assignable_extents.hpp` | `assignable_extents<From, To>` | Compile-time + runtime assignment compatibility check |
| `extents/convert_extents.hpp` | `ConvertExtents<From, To>::run(s)` | Convert between compatible extents types |
| `extents/extent_arithmetic.hpp` | `plus`, `minus`, `multiplies`, ... | Compile-time arithmetic on extent values with `dynamic_extent` propagation |
| `extents/extents_formatter.hpp` | `std::formatter<zipper::extents<...>>` | `std::format` support for extents objects |
| `extents/indices_in_range.hpp` | `indices_in_range(e, i...)` | Runtime bounds check |
| `extents/is_compatible.hpp` | `is_compatible<Sizes...>(e)` | Compile-time + runtime size compatibility check |
| `extents/is_cubic.hpp` | `is_cubic(e)`, `throw_if_not_cubic()` | Check / enforce all dimensions equal (e.g. square matrix) |
| `extents/offset_extents.hpp` | `OffsetExtents<From, Offset>` | Compute extents with dimensions shifted by an offset |

---

## Internal Detail (`detail/`)

| Path | Purpose |
|------|---------|
| `detail/dot.hpp` | Free-function dot product bypassing the `Form.hpp` dependency |
| `detail/tuple_to_array.hpp` | Convert STL tuple to `std::array` |
