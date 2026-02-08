#if !defined(ZIPPER_ZIPPERBASE_HPP)
#define ZIPPER_ZIPPERBASE_HPP

#include "concepts/Expression.hpp"
#include "concepts/IndexArgument.hpp"
#include "concepts/Zipper.hpp"
#include "expression/unary/Cast.hpp"
#include "expression/unary/CoefficientWiseOperation.hpp"
#include "expression/unary/concepts/ScalarOperation.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "expression/unary/Repeat.hpp"
#include "expression/unary/Slice.hpp"
#include "expression/unary/Swizzle.hpp"
#include "zipper/types.hpp"

namespace zipper {

template <template <concepts::QualifiedExpression> typename DerivedT,
          concepts::QualifiedExpression Expression>
class ZipperBase {
public:

  ZipperBase()
    requires(std::is_default_constructible_v<Expression>)
      : m_expression() {}
  using Derived = DerivedT<Expression>;
  auto derived() const -> const Derived & {
    return static_cast<const Derived &>(*this);
  }
  auto derived() -> Derived & { return static_cast<Derived &>(*this); }

  using expression_type = std::decay_t<Expression>;
  using expression_traits =
      expression::detail::ExpressionTraits<expression_type>;

  constexpr static bool is_const = std::is_const_v<Expression>;
  constexpr static bool is_writable =
      expression_traits::is_writable && !is_const;
  using value_type = typename expression_traits::value_type;
  using extents_type = typename expression_traits::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;

  auto expression() const -> const Expression & { return m_expression; }
  auto expression() -> Expression & { return m_expression; }
  auto extents() const -> const extents_type & {
    return expression().extents();
  }
  [[nodiscard]] constexpr auto extent(rank_type i) const -> index_type {
    return m_expression.extent(i);
  }
  static constexpr auto static_extent(rank_type i) -> index_type {
    return Expression::static_extent(i);
  }
  // template <typename... Args>
  // ZipperBase(Args&&... v) : m_expression(std::forward<Args>(v)...) {}

  ZipperBase(expression_type &&v) : m_expression(std::forward<Expression>(v)) {}
  ZipperBase(Derived &&v) : ZipperBase(std::move(v.expression())) {}
  ZipperBase(ZipperBase &&v) = default;

  ZipperBase(const Derived &v) : ZipperBase(v.expression()) {}
  ZipperBase(const expression_type &v) : m_expression(v) {}
  ZipperBase(const ZipperBase &v) = default;
  // Derived& operator=(concepts::ExpressionDerived auto const& v) {
  //     m_expression = v;
  //     return derived();
  // }

  operator value_type() const
    requires(extents_type::rank() == 0)
  {
    return (*this)();
  }

  template <concepts::Expression Other>
  ZipperBase(const Other &other)
    requires(is_writable && zipper::utils::extents::assignable_extents_v<
                                typename Other::extents_type, extents_type>)
      : m_expression(extents_traits::convert_from(other.extents())) {
    m_expression.assign(other);
  }

  template <typename... Args>
    requires(!(concepts::Zipper<Args> && ...))
  ZipperBase(Args &&...args) : m_expression(std::forward<Args>(args)...) {}

  template <concepts::Expression Other>
  auto operator=(const Other &other) -> Derived &
    requires(is_writable && zipper::utils::extents::assignable_extents_v<
                                typename Other::extents_type, extents_type>)
  {
    m_expression.assign(other);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
    return derived();
#pragma GCC diagnostic pop
  }
  template <concepts::Expression Other>
  auto operator=(Other &&other) -> Derived &
    requires(is_writable && zipper::utils::extents::assignable_extents_v<
                                typename std::decay_t<Other>::extents_type, extents_type>)
  {
    m_expression.assign(other);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
    return derived();
#pragma GCC diagnostic pop
  }

  template <concepts::Zipper Other>
  auto operator+=(const Other &other) -> Derived &
    requires(is_writable)
  {
    *this = *this + other;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
    return derived();
#pragma GCC diagnostic pop
  }
  template <concepts::Zipper Other>
  auto operator-=(const Other &other) -> Derived &
    requires(is_writable)
  {
    *this = *this - other;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
    return derived();
#pragma GCC diagnostic pop
  }
  auto operator*=(const value_type &other) -> Derived &
    requires(is_writable)
  {
    *this = other * *this;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
    return derived();
#pragma GCC diagnostic pop
  }
  auto operator/=(const value_type &other) -> Derived &
    requires(is_writable)
  {
    *this = *this / other;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
    return derived();
#pragma GCC diagnostic pop
  }

  template <typename OpType>
    requires(expression::unary::concepts::ScalarOperation<value_type, OpType>)
  auto unary_expr(const OpType &op) const {
    using V =
        expression::unary::CoefficientWiseOperation<const expression_type,
                                                     OpType>;
    return DerivedT<V>(V(expression(), op));
  }

  template <template <typename> typename BaseType = DerivedT,
            rank_type... ranks>
  auto swizzle() const {
    using V =
        expression::unary::Swizzle<const expression_type, ranks...>;
    return BaseType<V>(V(expression()));
  }
  template <typename T> auto cast() const {
    using V = expression::unary::Cast<T, const expression_type>;
    return DerivedT<V>(V(expression()));
  }

  template <concepts::IndexArgument... Args>
  auto operator()(Args &&...idxs) -> decltype(auto)
    requires(expression_traits::is_writable)
  {
    return expression()(
        filter_args_for_zipperbase(std::forward<Args>(idxs))...);
  }
  template <concepts::IndexArgument... Args>
  auto operator()(Args &&...idxs) const -> decltype(auto)

  {
    return expression()(
        filter_args_for_zipperbase(std::forward<Args>(idxs))...);
  }

  // pads left with dummy dimensions
  template <rank_type Count = 1,
            template <typename> typename BaseType = DerivedT>
  auto repeat_left() const {
    using V =
        expression::unary::Repeat<expression::unary::RepeatMode::Left,
                                  Count, const expression_type>;
    return BaseType<V>(V(expression()));
  }
  template <rank_type Count = 1,
            template <typename> typename BaseType = DerivedT>
  auto repeat_right() const {
    using V = expression::unary::Repeat<
        expression::unary::RepeatMode::Right, Count, const expression_type>;
    return BaseType<V>(V(expression()));
  }

protected:
  // slicing has fairly dimension specific effects for most derived types,
  // so we will just return the expression and let base class return things
  template <typename... Slices>
  auto slice_expression(Slices &&...slices) const {
    using my_expression_type =
        expression::unary::Slice<const expression_type,
                                 std::decay_t<Slices>...>;

    return my_expression_type(expression(),
                              filter_args_for_zipperbase(std::forward<Slices>(slices))...);
  }

  template <typename... Slices> auto slice_expression() const {
    using my_expression_type =
        expression::unary::Slice<const expression_type,
                                 std::decay_t<Slices>...>;
    return my_expression_type(expression(), Slices{}...);
  }

  template <typename... Slices>
  auto slice_expression(Slices &&...slices) {
    using my_expression_type =
        expression::unary::Slice<expression_type,
                                 std::decay_t<Slices>...>;
    return my_expression_type(expression(),
                              filter_args_for_zipperbase(std::forward<Slices>(slices))...);
  }
  template <typename... Slices> auto slice_expression() {
    using my_expression_type =
        expression::unary::Slice<expression_type,
                                 std::decay_t<Slices>...>;
    return my_expression_type(expression(), Slices{}...);
  }

public:
private:
  /// Helper to prevent zipperbase arguments get passed into the expression
  /// namespace
  template <typename T>
  static auto filter_args_for_zipperbase(T &&v) -> decltype(auto) {
    if constexpr (concepts::Zipper<std::decay_t<T>>) {
      return v.expression();
    } else {
      return std::forward<T>(v);
    }
  }
  Expression m_expression;
};

} // namespace zipper

#endif
