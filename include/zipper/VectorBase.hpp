#if !defined(ZIPPER_VECTORBASE_HPP)
#define ZIPPER_VECTORBASE_HPP

#include "ZipperBase.hpp"
#include "as.hpp"
#include "concepts/Vector.hpp"
#include "concepts/detail/IsZipperBase.hpp"
//
#include "ArrayBase.hpp"
#include "MatrixBase.hpp"
#include "detail/constexpr_arithmetic.hpp"
#include "detail/extents/constexpr_extent.hpp"
#include "expression/binary/CrossProduct.hpp"
#include "expression/reductions/CoefficientSum.hpp"
#include "expression/reductions/LpNorm.hpp"
#include "expression/reductions/LpNormPowered.hpp"
#include "expression/unary/Homogeneous.hpp"
#include "expression/nullary/MDSpan.hpp"

namespace zipper {
template <typename ValueType, index_type Rows> class Vector;

template <concepts::Expression Expr>
class VectorBase : public ZipperBase<VectorBase, Expr> {
public:
  VectorBase() = default;

  using Base = ZipperBase<VectorBase, Expr>;
  using expression_type = typename Base::expression_type;
  using expression_traits = typename Base::expression_traits;
  using value_type = typename expression_traits::value_type;
  using extents_type = typename expression_traits::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;
  static_assert(extents_traits::rank == 1);

  using Base::Base;
  // using Base::operator=;
  using Base::cast;
  using Base::extent;
  using Base::extents;
  using Base::expression;
  using Base::swizzle;

  auto eval() const { return Vector(*this); }
  template <concepts::Vector Other>
  VectorBase(const Other &other) : VectorBase(other.expression()) {}

  VectorBase(concepts::QualifiedExpression auto &v) : Base(v) {}
  VectorBase(concepts::QualifiedExpression auto &&v) : Base(std::move(v)) {}
  VectorBase(const extents_type &e) : Base(e) {}
  auto operator=(concepts::QualifiedExpression auto const &v) -> VectorBase & {
    return Base::operator=(v);
  }
  template <typename... Args>
  VectorBase(Args &&...args)
    requires(!(concepts::Vector<Args> && ...))
      : VectorBase(Expr(std::forward<Args>(args)...)) {}

  template <concepts::Vector Other>
  auto operator=(const Other &other) -> VectorBase &
    requires(expression_traits::is_writable)
  {
    expression().assign(other.expression());
    return *this;
  }
  template <concepts::Vector Other>
  auto operator=(Other &&other) -> VectorBase &
    requires(expression_traits::is_writable)
  {
    return operator=(other.expression());
  }
  [[nodiscard]] constexpr auto size() const -> index_type { return extent(0); }
  [[nodiscard]] constexpr auto rows() const -> index_type { return extent(0); }

  void resize(index_type size)
    requires(extents_traits::is_dynamic)
  {
    expression().resize(extents_type{size});
  }

  template <typename T>
  auto operator=(const std::initializer_list<T> &l) -> VectorBase &
    requires(extents_traits::is_static)
  {
    assert(l.size() == extent(0));
    for (index_type j = 0; j < extent(0); ++j) {
      (*this)(j) = std::data(l)[j];
    }
    return *this;
  }
  template <typename T>
  auto operator=(const std::initializer_list<T> &l) -> VectorBase &
    requires(extents_traits::is_dynamic)
  {
    if constexpr (expression_traits::is_resizable()) {
      expression().resize(extents_type(l.size()));
    }
    for (index_type j = 0; j < extent(0); ++j) {
      (*this)(j) = std::data(l)[j];
    }
    return *this;
  }

  auto as_array() const { return zipper::as_array(*this); }
  auto as_tensor() const { return zipper::as_tensor(*this); }
  auto as_form() const { return zipper::as_form(*this); }

  template <index_type T> auto norm_powered() const -> value_type {
    return expression::reductions::LpNormPowered<T, const expression_type>(expression())();
  }
  auto norm_powered(value_type T) const -> value_type {
    return expression::reductions::CoefficientSum{as_array().pow(T).abs().expression()}();
  }

  auto dot(concepts::Vector auto const &o) const -> value_type {
    return as_form() * o;
  }
  template <concepts::Vector O> auto cross(O const &o) const {
    using V = expression::binary::CrossProduct<const expression_type,
                                               const typename O::expression_type>;
    return VectorBase<V>(V(expression(), o.expression()));
  }

  template <index_type Start, index_type Size>
  auto segment()
    requires(expression_traits::is_writable)
  {
    auto S = slice(std::integral_constant<index_type, Start>{},
                   std::integral_constant<index_type, Size>{});
    auto v = Base::slice_expression(S);
    using V = std::decay_t<decltype(v)>;
    return VectorBase<V>(std::move(v));
  }
  template <index_type Start, index_type Size> auto segment() const {
    auto S = slice(std::integral_constant<index_type, Start>{},
                   std::integral_constant<index_type, Size>{});
    auto v = Base::slice_expression(S);
    using V = std::decay_t<decltype(v)>;
    return VectorBase<V>(std::move(v));
  }
  template <index_type Size>
  auto segment(index_type start)
    requires(expression_traits::is_writable)
  {
    auto S = slice(start, std::integral_constant<index_type, Size>{});
    auto v = Base::slice_expression(S);
    using V = std::decay_t<decltype(v)>;
    return VectorBase<V>(std::move(v));
  }
  template <index_type Size> auto segment(index_type start) const {
    auto S = slice(start, Size);
    auto v = Base::slice_expression(S);
    using V = std::decay_t<decltype(v)>;
    return VectorBase<V>(std::move(v));
  }
  auto segment(index_type start, index_type size)
    requires(expression_traits::is_writable)
  {
    auto S = slice(start, size);
    auto v = Base::slice_expression(S);
    using V = std::decay_t<decltype(v)>;
    return VectorBase<V>(std::move(v));
  }
  auto segment(index_type start, index_type size) const {
    auto S = slice(start, size);
    auto v = Base::slice_expression(S);
    using V = std::decay_t<decltype(v)>;
    return VectorBase<V>(std::move(v));
  }

  template <index_type I>
  auto head()
    requires(expression_traits::is_writable)
  {
    auto S = slice(std::integral_constant<index_type, 0>{},
                   std::integral_constant<index_type, I>{});
    auto v = Base::slice_expression(S);
    using V = std::decay_t<decltype(v)>;
    return VectorBase<V>(std::move(v));
  }
  template <index_type I> auto head() const {
    auto S = slice(std::integral_constant<index_type, 0>{},
                   std::integral_constant<index_type, I>{});
    auto v = Base::slice_expression(S);
    using V = std::decay_t<decltype(v)>;
    return VectorBase<V>(std::move(v));
  }
  auto head(index_type N)
    requires(expression_traits::is_writable)
  {
    auto S = slice<std::integral_constant<index_type, 0>, index_type>(
        std::integral_constant<index_type, 0>{}, N);
    auto v = Base::slice_expression(S);
    using V = std::decay_t<decltype(v)>;
    return VectorBase<V>(std::move(v));
  }
  auto head(index_type N) const {
    auto S = slice(std::integral_constant<index_type, 0>{}, N);
    auto v = Base::slice_expression(S);
    using V = std::decay_t<decltype(v)>;
    return VectorBase<V>(std::move(v));
  }

  auto get_tail_slice(index_type I) const {
    return slice(
        detail::minus(detail::extents::constexpr_extent<0>(extents()), I), I);
  }
  template <index_type I> auto get_tail_slice() const {
    return slice(detail::minus(detail::extents::constexpr_extent<0>(extents()),
                               std::integral_constant<index_type, I>{}),
                 std::integral_constant<index_type, I>{});
  }
  template <index_type I>
  auto tail()
    requires(expression_traits::is_writable)
  {
    auto S = get_tail_slice<I>();
    auto v = Base::slice_expression(S);
    using V = std::decay_t<decltype(v)>;
    return VectorBase<V>(std::move(v));
  }
  template <index_type I> auto tail() const {
    auto S = get_tail_slice<I>();
    auto v = Base::slice_expression(S);
    using V = std::decay_t<decltype(v)>;
    return VectorBase<V>(std::move(v));
  }
  auto tail(index_type N)
    requires(expression_traits::is_writable)
  {
    auto S = get_tail_slice(N);
    auto v = Base::slice_expression(S);
    using V = std::decay_t<decltype(v)>;
    return VectorBase<V>(std::move(v));
  }
  auto tail(index_type N) const {
    auto S = get_tail_slice(N);
    auto v = Base::slice_expression(S);
    using V = std::decay_t<decltype(v)>;
    return VectorBase<V>(std::move(v));
  }

  // implements ones * this.transpose()
  auto repeat_left() const {
    return Base::template repeat_left<1, MatrixBase>();
  }
  // implements  this * ones.transpose()
  auto repeat_right() const {
    return Base::template repeat_right<1, MatrixBase>();
  }

  template <index_type T = 2> auto norm() const -> value_type {
    return expression::reductions::LpNorm<T, const expression_type>(expression())();
  }
  auto norm(value_type T) const -> value_type {
    value_type p = value_type(1.0) / T;
    return std::pow<value_type>(norm_powered(T), p);
  }

  template <index_type T = 2> auto normalized() const {
    return *this / norm<T>();
  }
  auto normalized(value_type T) const -> value_type { return *this / norm(T); }
  template <index_type T = 2> void normalize() { *this /= norm<T>(); }
  void normalize(value_type T) { *this /= norm(T); }

  template <expression::unary::HomogeneousMode Mode =
                expression::unary::HomogeneousMode::Position>
  auto homogeneous() const {
    return VectorBase<expression::unary::Homogeneous<Mode, const expression_type>>(expression());
  }
};

template <concepts::Expression Expr>
VectorBase(Expr &&) -> VectorBase<Expr>;
template <concepts::Expression Expr>
VectorBase(const Expr &) -> VectorBase<Expr>;

// Deduction guides from std::span
template <typename T, std::size_t N>
VectorBase(std::span<T, N>) -> VectorBase<
    expression::nullary::MDSpan<T, extents<N>>>;

template <typename T>
VectorBase(std::span<T, std::dynamic_extent>) -> VectorBase<
    expression::nullary::MDSpan<T, extents<dynamic_extent>>>;

// Deduction guides from std::vector (creates a mutable dynamic-extent view)
template <typename T, typename Alloc>
VectorBase(std::vector<T, Alloc> &) -> VectorBase<
    expression::nullary::MDSpan<T, extents<dynamic_extent>>>;

template <typename T, typename Alloc>
VectorBase(const std::vector<T, Alloc> &) -> VectorBase<
    expression::nullary::MDSpan<const T, extents<dynamic_extent>>>;

// Deduction guides from std::array (creates a view with static extent)
template <typename T, std::size_t N>
VectorBase(std::array<T, N> &) -> VectorBase<
    expression::nullary::MDSpan<T, extents<N>>>;

template <typename T, std::size_t N>
VectorBase(const std::array<T, N> &) -> VectorBase<
    expression::nullary::MDSpan<const T, extents<N>>>;

UNARY_DECLARATION(VectorBase, LogicalNot, operator!)
UNARY_DECLARATION(VectorBase, BitNot, operator~)
UNARY_DECLARATION(VectorBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(VectorBase, Divides, operator/)

BINARY_DECLARATION(VectorBase, Plus, operator+)
BINARY_DECLARATION(VectorBase, Minus, operator-)

template <concepts::Vector Expr1, concepts::Vector Expr2>
auto operator==(Expr1 const &lhs, Expr2 const &rhs) -> bool {
  return (lhs.as_array() == rhs.as_array()).all();
}

template <concepts::Vector Expr1, concepts::Vector Expr2>
auto operator!=(Expr1 const &lhs, Expr2 const &rhs) -> bool {
  return (lhs.as_array() != rhs.as_array()).any();
}
template <concepts::Vector Expr>
auto operator*(Expr const &lhs, typename Expr::value_type const &rhs) {
  using V =
      expression::unary::ScalarMultiplies<typename Expr::value_type,
                                         const typename Expr::expression_type, true>;
  return VectorBase<V>(V(lhs.expression(), rhs));
}
template <concepts::Vector Expr>
auto operator*(typename Expr::value_type const &lhs, Expr const &rhs) {
  using V =
      expression::unary::ScalarMultiplies<typename Expr::value_type,
                                         const typename Expr::expression_type, false>;
  return VectorBase<V>(V(lhs, rhs.expression()));
}

namespace concepts::detail {

template <typename T>
struct IsVector<VectorBase<T>> : std::true_type {};
template <typename T>
struct IsZipperBase<VectorBase<T>> : std::true_type {};
} // namespace concepts::detail

} // namespace zipper
#endif
