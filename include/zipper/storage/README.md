# Storage

The storage layer provides the raw data containers that back nullary
expressions. Storage types are decoupled from the expression/index-mapping
layer -- they provide flat linear access, while `LinearLayoutExpression`
handles the multidimensional-to-linear mapping.

---

## Dense Data

| Header | Description |
|--------|-------------|
| `DenseData.hpp` | Dispatch header: selects static or dynamic specialisation |
| `StaticDenseData.hpp` | `DenseData<T, N>` for `N > 0`: backed by `std::array<T, N>` |
| `DynamicDenseData.hpp` | `DenseData<T, dynamic_extent>`: backed by `std::vector<T>` |

`DenseData` provides:
- `coeff(i)`, `coeff_ref(i)`, `const_coeff_ref(i)` -- element access
- `data()` -- raw pointer
- `begin()` / `end()` -- iterators
- `as_std_span()` -- span view
- `resize(n)` (dynamic only)

## Span Data

| Header | Description |
|--------|-------------|
| `SpanData.hpp` | Non-owning storage backed by `std::span<T>`; used by `MDSpan` |

## Sparse Data

| Header | Description |
|--------|-------------|
| `SparseCoordinateAccessor.hpp` | COO-format sparse accessor |
| `detail/SparseCompressedData.hpp` | CSR/CSC compressed sparse data |
| `detail/to_sparse_compressed_data.hpp` | Conversion to compressed format |

## Layout

| Header | Description |
|--------|-------------|
| `layout_types.hpp` | `tensor_layout<LeftMajor>` -- selects `layout_left` (column-major) or `layout_right` (row-major) |

## Traits and Concepts

| Header | Description |
|--------|-------------|
| `LinearAccessorTraits.hpp` | `AccessFeatures` and `ShapeFeatures` structs; `BasicLinearAccessorTraits` base |
| `StlStorageInfo.hpp` | Detects STL container properties (static/dynamic size, value_type) |
| `concepts/Data.hpp` | `Data` concept |
| `concepts/DenseData.hpp` | `DenseData` concept |
| `concepts/StaticData.hpp` | `StaticData` concept |
| `concepts/DynamicData.hpp` | `DynamicData` concept |
| `concepts/Accessor.hpp` | `Accessor` concept |
