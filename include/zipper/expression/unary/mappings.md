# Mappings

Zipper supports a variety of mappings between different tensor types.
That is, zipper allows for broadcasting to a new axis like how `meshgrid` maps
1-tensors to 2-tensors:

```cpp
zipper::Vector<double, 3> x{0.0, 0.5, 1.0};
zipper::Vector<double, 4> y{0.0, 1.0, 2.0, 3.0};

// X(i, j) = x(i), Y(i, j) = y(j): NumPy's meshgrid(x, y, indexing="ij")
zipper::Matrix X = x.broadcast<1>(zipper::extents<3, 4>{});  // 3x4
zipper::Matrix Y = y.broadcast<0>(zipper::extents<3, 4>{});  // 3x4
```

The template arguments of `broadcast` are the positions of the new axes in the
result, whose extents are given; the remaining axes are the original ones, in
order.

It also supports reductions like Eigen's `A.colwise().norm()` via the same
syntax

```cpp
zipper::Matrix<double,3,100> A;
zipper::Vector x = A.colwise().norm();
zipper::Vector y = A.over<0>().norm();  // the same: colwise() is over<0>()
```

`over<Axes...>()` selects the axes each sub-tensor spans; reductions (`sum`,
`min`, `max`, `norm`, `contract`) replace each sub-tensor with a scalar, so
those axes disappear.

This is done via a generic `map` interface that lets you select a set of input
axes, output axes, and a functor:

```cpp
zipper::Tensor<double,40,20,5,5> T = {};

// D is a 5x40x20 tensor whose first axis is the diagonals of each 5x5 matrix
// We could have done (auto const& A) as well
zipper::Tensor D = T.over<2,3>().map<0>(
  [](const zipper::concepts::MatrixExpression auto &A) {
  return zipper::MatrixBase(A).diagonal();
});

```

## Map Functors

Map functors "just" need to accept a zipper expression of the right rank and
return a result whose shape is the same for every sub-tensor (see below for
what they may return).
An idiosyncrasy of this powerful mechanism, implemented agnostic of algebraic
structure, is that the mapping structure does not forward the inner or outer
types. This means that even though Eigen might have treated `M.rowwise()` as a
reducer of each row as vectors, `M.rowwise().map(f)` will just feed `f` a zipper
expression (the underlying object in `Matrix`/`FormBase`/`Vector`/etc.). This lets
users pick what type of structure they want to read this as - this means `rowwise()`
can be used with its row-vectors as 1-forms rather than as col-vectors.

A simple matrix-vector can therefore be written as

```cpp
zipper::Matrix<...> A = {};
zipper::Vector<...> x = {};

zipper::Vector b = A.rowwise().map([&](auto const& r) {
  return zipper::FormBase(r) * x;
});
```

As previously shown, you can check for errors / constrain the functor's
parameter by what it needs (`zipper::concepts::MatrixExpression auto`,
`zipper::concepts::RankedExpression<T, N>`) and then a quick line to wrap the
input expressions into the type you want (`zipper::MatrixBase(A)`,
`zipper::Vector x = ...`, `zipper::Tensor D = ...`) or a conversion
(`zipper::as_array(e)`, `zipper::as_vector(e)`, ...).

The functor returns a scalar, an expression or zipper object, or a callable
taking the sub-tensor's indices; the axes of its result go to the positions
given to `map` (axis 0 in the diagonal example), and the remaining axes of `T`
follow in order.
Without positions, a scalar result removes the selected axes and a result
shaped like the sub-tensor stays in place.

These examples are compiled and checked by `tests/mappings_examples.cpp`.
