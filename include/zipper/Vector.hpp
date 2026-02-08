#if !defined(ZIPPER_VECTOR_HPP)
#define ZIPPER_VECTOR_HPP

#include "VectorBase.hpp"
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
  Vector &operator=(Vector &&o) {
    expression().operator=(std::move(o.expression()));
    return *this;
  }
  Vector(index_type size)
    requires(extents_traits::is_dynamic)
      : Base(zipper::extents<Rows>(size)) {}

#if defined(NDEBUG)
  Vector(index_type)
#else
  Vector(index_type rows)
#endif
    requires(extents_traits::is_static)
      : Base() {
    assert(rows == extent(0));
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
    assert(l.size() == extent(0));
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
