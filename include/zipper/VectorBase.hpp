#if !defined(ZIPPER_VECTORBASE_HPP)
#define ZIPPER_VECTORBASE_HPP

#include "ZipperBase.hpp"
#include "as.hpp"
#include "concepts/Vector.hpp"
#include "concepts/detail/IsZipperBase.hpp"
#include "concepts/stl.hpp"
#include "detail/assert.hpp"
//
#include "ArrayBase.hpp"
#include "MatrixBase.hpp"
#include "detail/constexpr_arithmetic.hpp"
#include "detail/extents/constexpr_extent.hpp"
#include "expression/binary/CrossProduct.hpp"
#include "expression/nullary/StlMDArray.hpp"
#include "expression/reductions/CoefficientSum.hpp"
#include "expression/reductions/LpNorm.hpp"
#include "expression/reductions/LpNormPowered.hpp"
#include "expression/unary/Homogeneous.hpp"

namespace zipper {
template <typename ValueType, index_type Rows> class Vector;

template <concepts::Expression Expr>
  requires(concepts::QualifiedRankedExpression<Expr, 1>)
class VectorBase : public ZipperBase<VectorBase, Expr> {
public:
  VectorBase() = default;

  using Base = ZipperBase<VectorBase, Expr>;
  using expression_type = typename Base::expression_type;
  using expression_traits = typename Base::expression_traits;
  using value_type = typename expression_traits::value_type;
  using extents_type = typename expression_traits::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;

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

  VectorBase(concepts::QualifiedExpression auto &v)
    requires(expression::concepts::OwningExpression<expression_type>)
      : Base(v) {}
  VectorBase(concepts::QualifiedExpression auto &&v)
    requires(expression::concepts::OwningExpression<expression_type>)
      : Base(std::move(v)) {}
  VectorBase(const extents_type &e) : Base(e) {}
  auto operator=(concepts::QualifiedExpression auto const &v) -> VectorBase & {
    return Base::operator=(v);
  }
  template <typename... Args>
    requires(!(concepts::Vector<Args> && ...))
  VectorBase(Args &&...args)
      : Base(std::in_place, std::forward<Args>(args)...) {}

  template <concepts::Vector Other>
  auto operator=(const Other &other) -> VectorBase &
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    expression().assign(other.expression());
    return *this;
  }
  template <concepts::Vector Other>
  auto operator=(Other &&other) -> VectorBase &
    requires(expression::concepts::WritableExpression<expression_type>)
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
    ZIPPER_ASSERT(l.size() == extent(0));
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
    return expression::reductions::LpNormPowered<T, const expression_type &>(expression())();
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
    return VectorBase<V>(std::in_place, expression(), o.expression());
  }

  template <index_type Start, index_type Size>
  auto segment()
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    auto S = slice(std::integral_constant<index_type, Start>{},
                   std::integral_constant<index_type, Size>{});
    using V = expression::unary::Slice<expression_type&, std::decay_t<decltype(S)>>;
    return VectorBase<V>(std::in_place, expression(), S);
  }
  template <index_type Start, index_type Size> auto segment() const {
    auto S = slice(std::integral_constant<index_type, Start>{},
                   std::integral_constant<index_type, Size>{});
    using V = expression::unary::Slice<const expression_type&, std::decay_t<decltype(S)>>;
    return VectorBase<V>(std::in_place, expression(), S);
  }
  template <index_type Size>
  auto segment(index_type start)
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    auto S = slice(start, std::integral_constant<index_type, Size>{});
    using V = expression::unary::Slice<expression_type&, std::decay_t<decltype(S)>>;
    return VectorBase<V>(std::in_place, expression(), S);
  }
  template <index_type Size> auto segment(index_type start) const {
    auto S = slice(start, Size);
    using V = expression::unary::Slice<const expression_type&, std::decay_t<decltype(S)>>;
    return VectorBase<V>(std::in_place, expression(), S);
  }
  auto segment(index_type start, index_type size)
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    auto S = slice(start, size);
    using V = expression::unary::Slice<expression_type&, std::decay_t<decltype(S)>>;
    return VectorBase<V>(std::in_place, expression(), S);
  }
  auto segment(index_type start, index_type size) const {
    auto S = slice(start, size);
    using V = expression::unary::Slice<const expression_type&, std::decay_t<decltype(S)>>;
    return VectorBase<V>(std::in_place, expression(), S);
  }

  template <index_type I>
  auto head()
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    auto S = slice(std::integral_constant<index_type, 0>{},
                   std::integral_constant<index_type, I>{});
    using V = expression::unary::Slice<expression_type&, std::decay_t<decltype(S)>>;
    return VectorBase<V>(std::in_place, expression(), S);
  }
  template <index_type I> auto head() const {
    auto S = slice(std::integral_constant<index_type, 0>{},
                   std::integral_constant<index_type, I>{});
    using V = expression::unary::Slice<const expression_type&, std::decay_t<decltype(S)>>;
    return VectorBase<V>(std::in_place, expression(), S);
  }
  auto head(index_type N)
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    auto S = slice<std::integral_constant<index_type, 0>, index_type>(
        std::integral_constant<index_type, 0>{}, N);
    using V = expression::unary::Slice<expression_type&, std::decay_t<decltype(S)>>;
    return VectorBase<V>(std::in_place, expression(), S);
  }
  auto head(index_type N) const {
    auto S = slice(std::integral_constant<index_type, 0>{}, N);
    using V = expression::unary::Slice<const expression_type&, std::decay_t<decltype(S)>>;
    return VectorBase<V>(std::in_place, expression(), S);
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
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    auto S = get_tail_slice<I>();
    using V = expression::unary::Slice<expression_type&, std::decay_t<decltype(S)>>;
    return VectorBase<V>(std::in_place, expression(), S);
  }
  template <index_type I> auto tail() const {
    auto S = get_tail_slice<I>();
    using V = expression::unary::Slice<const expression_type&, std::decay_t<decltype(S)>>;
    return VectorBase<V>(std::in_place, expression(), S);
  }
  auto tail(index_type N)
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    auto S = get_tail_slice(N);
    using V = expression::unary::Slice<expression_type&, std::decay_t<decltype(S)>>;
    return VectorBase<V>(std::in_place, expression(), S);
  }
  auto tail(index_type N) const {
    auto S = get_tail_slice(N);
    using V = expression::unary::Slice<const expression_type&, std::decay_t<decltype(S)>>;
    return VectorBase<V>(std::in_place, expression(), S);
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
    return expression::reductions::LpNorm<T, const expression_type &>(expression())();
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
    return VectorBase<expression::unary::Homogeneous<Mode, const expression_type&>>(std::in_place, expression());
  }
};

template <concepts::Expression Expr>
VectorBase(Expr &&) -> VectorBase<Expr>;
template <concepts::Expression Expr>
VectorBase(const Expr &) -> VectorBase<Expr>;

// STL deduction guides: rvalue → owning StlMDArray, lvalue → borrowing StlMDArray
template <concepts::StlStorageOfRank<1> S>
VectorBase(S &&) -> VectorBase<expression::nullary::StlMDArray<std::decay_t<S>>>;
template <concepts::StlStorageOfRank<1> S>
VectorBase(S &) -> VectorBase<expression::nullary::StlMDArray<S &>>;

namespace concepts::detail {

template <typename T>
struct IsVector<VectorBase<T>> : std::true_type {};
template <typename T>
struct IsZipperBase<VectorBase<T>> : std::true_type {};
} // namespace concepts::detail

} // namespace zipper
#endif
