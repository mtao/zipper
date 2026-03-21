# Concepts

C++20 concepts that define the type taxonomy in Zipper. These are used
throughout the library for constraining constructors, operators, and
template parameters.

---

## Type Hierarchy Concepts

These form a refinement hierarchy mirroring the CRTP wrapper classes:

```
Expression              any expression node
  Zipper                any wrapper (ZipperBase<Derived, Expr>)
    DataArray           plain data storage (no arithmetic)
      Tensor            rank-N tensor (generic)
        Array           coefficient-wise operations
        Matrix          rank-2 matrix algebra
        Vector          rank-1 column vector
        Form            rank-1 row vector / covector
```

| Header | Concept | Satisfied By |
|--------|---------|--------------|
| `Expression.hpp` | `Expression<T>`, `QualifiedExpression<T>` | All expression nodes; `Qualified` accepts const/ref-qualified types |
| `Zipper.hpp` | `Zipper<T>` | All `ZipperBase<...>` wrappers |
| `DataArray.hpp` | `DataArray<T>` | `DataArrayBase<Expr>`, `DataArray_<...>` |
| `Tensor.hpp` | `Tensor<T>` | `TensorBase<Expr>` and subclasses |
| `Array.hpp` | `Array<T>` | `ArrayBase<Expr>` |
| `Matrix.hpp` | `Matrix<T>` | `MatrixBase<Expr>` |
| `Vector.hpp` | `Vector<T>` | `VectorBase<Expr>` |
| `Form.hpp` | `Form<T>` | `FormBase<Expr>` |

**Important:** `Zipper` and `Expression` are **disjoint** -- wrapper types
satisfy `Zipper` but not `Expression`, and vice versa. To get the underlying
expression from a wrapper, use `.expression()`.

---

## Index and Shape Concepts

| Header | Concept | Description |
|--------|---------|-------------|
| `Extents.hpp` | `Extents<T>` | Any `std::extents<...>` type |
| `Index.hpp` | `Index<T>` | Valid index type (integral or `integral_constant`) |
| `IndexArgument.hpp` | `IndexArgument<T>` | Valid argument to `operator()` |
| `IndexSlice.hpp` | `IndexSlice<T>` | Valid argument to `.slice()` |
| `shapes.hpp` | `SquareExtents<E>`, `VectorExtents<E>`, `MatrixExtents<E>` | Shape predicates on extents types |

---

## Interoperability Concepts

| Header | Concept | Description |
|--------|---------|-------------|
| `stl.hpp` | `StlStaticStorage<T>`, `StlDynamicStorage<T>` | Recognises `std::array`, `std::vector`, etc. for deduction-guide construction |
| `Preconditioner.hpp` | `Preconditioner<T>` | Types that can precondition a linear system |
| `DirectSolver.hpp` | `DirectSolver<T>` | Types that can directly solve a linear system |

---

## Aggregate Header

| Header | Description |
|--------|-------------|
| `all.hpp` | Includes all concept headers for convenience |

---

## Implementation Detail

| Header | Description |
|--------|-------------|
| `detail/IsZipperBase.hpp` | SFINAE trait `IsZipperBase<T>` / `IsDataArray<T>` etc.; specialised by each wrapper class to opt into the concept hierarchy |

Each wrapper class (e.g., `DataArrayBase<Expr>`) provides a partial
specialisation of `IsDataArray` in `concepts::detail` to register itself.
The concepts then check `concepts::detail::IsDataArray<std::decay_t<T>>::value`.
