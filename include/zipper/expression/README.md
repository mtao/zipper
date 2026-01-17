# zipper::expression

<!--toc:start-->
- [zipper::expression](#zipperexpression)
  - [`MDArray`](#mdarray)
  - [Expression](#expression)
<!--toc:end-->

The central object in zipper is an _expression_ over multi-dimensional arrays
(`MDArray`s), where a MDArray has no built-in concepts like addition or
multiplication. Classes like `zipper::MatrixView`, `zipper::FormView`, or
`zipper::TensorView` will store a single expression via a class derived from
`zipper::expresion::ExpressionBase` as a member, and operators between these
classes classes interact a new `zipper::*View` will be created a new
`Expression` type, representing the expression of those respective operators.
generated.

## `MDArray`

A `MDArray` is a multi-dimensional array, i.e an element of
$\mathcal{F}^{e_0 \times e_1, \ldots, e_n}$ for some field $\mathcal{F}$ and
some set of $n$ extents. For example:

- $\mathbb{R}^0$ is the set of scalar real values,
- $\mathbb{Z}^3$ is the set of single dimensional arrays of integers with $3$
elements,
- $\mathbb{Z_2}^{2\times2}$ is the set of square matrices with width $2$ with
boolean values.

A `MDArray` satisfies a basic interface:

- A `consteval` function `rank() -> rank_type` that is the number of dimensions
  in the array.
- A `extent(rank_type d) -> index_type` that is the size of the array in the
dimension `d`. An extent of $0$ will be interpretted as $\infty$.
- Value access to elements via `coeff(index_type... i) const -> value_type`
- Reference access to internal data via `coeff_ref(index_type... i) -> value_type&`
- `const` Reference access to internal data via `const_coeff_ref(index_type... i) const -> value_type&`

For convenience bracket access is also allowed via `operator()`

## Expression

An _Operation_ is any a map from $0$ or more `MDArray`s to `MDArray`s. That is,
an expression is a function space of `MDArray`s. In this library we lazily
evaluate _Operation_s by building trees of operations without evaluating them,
i.e _Expression_ trees.
In this library we use the Curiously Recurring Template Patterm (CRTP) and
expression templates to represent to statically store the expressions.
The evaluation of an `Expression` is always a `MDArray`, and so each expression
derives its extents. Every `Expression` has to satisfy

- `rank() -> rank_type`
- `extent(rank_type d) -> index_type`
- `coeff(index_type... i) const -> value_type`
In general `Expression`s cannot be written to
and if the expression has the `has_plain_data` trait then
- `coeff_ref(index_type... i) const -> value_type&`
- `const_coeff_ref(index_type... i) const -> value_type const&`
