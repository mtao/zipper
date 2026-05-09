#if !defined(ZIPPER_VECTOR_HPP)
#define ZIPPER_VECTOR_HPP

/// @file Vector.hpp
/// @brief Owning dense column vector type with static or dynamic extent.
/// @ingroup user_types
///
/// `Vector<T, Rows>` is the primary user-facing column vector type.  It owns
/// its data via an `MDArray` storage backend and inherits the full `VectorBase`
/// interface (arithmetic operators, dot product, norm, cross product, etc.).
///
/// Template parameters:
///   - `T`: scalar type (e.g. `double`, `float`).
///   - `Rows`: number of elements (`dynamic_extent` for runtime-sized).
///
/// Construction:
///   - Default: `Vector<double, 3> v;` (static, zero-initialised).
///   - From initializer list: `Vector<double, 3> v({1.0, 2.0, 3.0});`
///   - Dynamic: `Vector<double, dynamic_extent> v(n);`
///   - Copy from expression: `Vector<double, 3> v(some_expression);`
///
/// @code
///   // Static 3-vector
///   Vector<double, 3> v({1.0, 2.0, 3.0});
///
///   // Dynamic vector
///   Vector<double, dynamic_extent> w(100);
///
///   // Vector arithmetic
///   auto sum = v + w;          // coefficient-wise addition
///   double d = v.dot(w);       // dot product
///   double n = v.norm();       // Euclidean norm
///   auto c = v.cross(w);       // cross product (3D only)
///
///   // Iteration
///   for (auto& x : v) { x *= 2.0; }
///
///   // Span views (non-owning)
///   auto s = v.as_span();
/// @endcode
///
/// The type alias `VectorX<T>` is provided for dynamic-extent vectors:
///   `VectorX<double>` is equivalent to `Vector<double, dynamic_extent>`.
///
/// @see zipper::VectorBase — CRTP base providing the vector interface.
/// @see zipper::Matrix — owning matrix type.
/// @see zipper::Form — owning row vector (1-form) type.
/// @see zipper::expression::nullary::MDArray — the underlying owning storage.
/// @see zipper::expression::nullary::Unit — unit (basis) vector expression.
/// @see zipper::utils::solver — iterative solvers that produce Vector results.
/// @see zipper::utils::solver::SolverResult — solver result containing a
///      Vector solution.

#include "VectorBase.hxx"
#include "zipper/detail/assert.hpp"
#include "zipper/detail/extents_check.hpp"
#include "zipper/expression/nullary/MDArray.hpp"
#include "zipper/expression/nullary/MDSpan.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, index_type Rows>
class Vector
    : public VectorBase<expression::nullary::MDArray<ValueType, zipper::extents<Rows>>> {
public:
  using expression_type =
      expression::nullary::MDArray<ValueType, zipper::extents<Rows>>;
  using Base = VectorBase<expression_type>;
  using Base::expression;
  using value_type = Base::value_type;
  using extents_type = Base::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;
  using Base::extent;
  using Base::extents;
  constexpr static bool is_static = extents_traits::is_static;

  using span_expression_type =
      expression::nullary::MDSpan<ValueType, zipper::extents<Rows>>;
  using const_span_expression_type =
      expression::nullary::MDSpan<const ValueType, zipper::extents<Rows>>;
  using span_type = VectorBase<span_expression_type>;
  using const_span_type = VectorBase<const_span_expression_type>;

  Vector() = default;
  Vector(const Vector &o) = default;
  Vector &operator=(const Vector &o) = default;
  Vector(Vector &&o) = default;
  Vector &operator=(Vector &&o) = default;
  Vector(index_type size)
    requires(extents_traits::is_dynamic)
      : Base(zipper::extents<Rows>(size)) {}

  Vector([[maybe_unused]] index_type rows)
    requires(extents_traits::is_static)
      : Base() {
    detail::check_extents<extents_type>(rows);
  }

  template <index_type R2>
  Vector(const Vector<value_type, R2> &other) : Base(other.expression()) {}

  template <concepts::Vector Other>
  Vector(const Other &other) : Base(other) {}

  template <concepts::Expression Other>
  Vector(const Other &other) : Base(other) {}

  template <typename T>
  Vector(const std::initializer_list<T> &l)
    requires(extents_traits::is_static)
  {
    detail::check_extents<extents_type>(l.size());
    std::ranges::copy(l, begin());
  }
  template <typename T>
  Vector(const std::initializer_list<T> &l)
    requires(extents_traits::is_dynamic)
      : Base(extents_type(l.size())) {
    std::ranges::copy(l, begin());
  }

  span_type as_span() {
    if constexpr (is_static) {
      return span_type(expression().as_std_span());
    } else {
      return span_type(expression().as_std_span(), extents());
    }
  }
  const_span_type as_const_span() const {
    if constexpr (is_static) {
      return const_span_type(expression().as_std_span());
    } else {
      return const_span_type(expression().as_std_span(), extents());
    }
  }
  const_span_type as_span() const { return as_const_span(); }

  auto begin() { return expression().linear_accessor().begin(); }
  auto end() { return expression().linear_accessor().end(); }
  auto begin() const { return expression().linear_accessor().begin(); }
  auto end() const { return expression().linear_accessor().end(); }

  using Base::operator=;
};

template <typename T> using VectorX = Vector<T, dynamic_extent>;

template <concepts::Vector MB>
Vector(const MB &o) -> Vector<std::decay_t<typename MB::value_type>,
                              MB::extents_type::static_extent(0)>;
template <concepts::Expression MB>
  requires(MB::extents_type::rank() == 1)
Vector(const MB &o) -> Vector<std::decay_t<typename MB::value_type>,
                              MB::extents_type::static_extent(0)>;

namespace concepts::detail {
template <typename ValueType, index_type Rows>
struct IsVector<zipper::Vector<ValueType, Rows>> : std::true_type {};
template <typename ValueType, index_type Rows>
struct IsZipperBase<zipper::Vector<ValueType, Rows>> : std::true_type {};
} // namespace concepts::detail
} // namespace zipper

#endif
