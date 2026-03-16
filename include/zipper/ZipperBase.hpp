#if !defined(ZIPPER_ZIPPERBASE_HPP)
#define ZIPPER_ZIPPERBASE_HPP

#include "concepts/Expression.hpp"
#include "concepts/IndexArgument.hpp"
#include "concepts/Zipper.hpp"
#include "detail/NonReturnable.hpp"
#include "expression/concepts/capabilities.hpp"
#include "expression/unary/Cast.hpp"
#include "expression/unary/CoefficientWiseOperation.hpp"
#include "expression/unary/Diagonal.hpp"
#include "expression/unary/Repeat.hpp"
#include "expression/unary/Slice.hpp"
#include "expression/unary/Swizzle.hpp"
#include "expression/unary/UnsafeRef.hpp"
#include "expression/unary/concepts/ScalarOperation.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/types.hpp"

#include <utility> // std::in_place_t, std::in_place

namespace zipper {

template <concepts::Expression Expr>
  requires(concepts::QualifiedRankedExpression<Expr, 1>)
class VectorBase;

namespace detail {
/// Maps Zipper wrapper types to their underlying expression_type so that
/// Slice template parameters always use raw expression types (which are
/// freely copyable) rather than wrapper types (which may be NonReturnable).
template <typename T>
struct slice_type_for {
    using type = T;
};
template <typename T>
    requires(concepts::Zipper<T>)
struct slice_type_for<T> {
    using type = typename T::expression_type;
};
template <typename T>
using slice_type_for_t = typename slice_type_for<T>::type;
}  // namespace detail

template <template <typename> typename DerivedT,
          concepts::QualifiedExpression Expression>
class ZipperBase
    : public detail::returnability_mixin_t<
          expression::detail::ExpressionTraits<
              std::decay_t<Expression>>::stores_references ||
          std::is_reference_v<Expression>> {
public:
  ZipperBase()
    requires(std::is_default_constructible_v<Expression>)
      : m_expression() {}
  using Derived = DerivedT<Expression>;
  auto derived(this auto &self) -> auto & {
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(self)>>) {
      return static_cast<const Derived &>(self);
    } else {
      return static_cast<Derived &>(self);
    }
  }

  using raw_expression_type =
      Expression; // the template parameter, possibly const/ref qualified
  using expression_type = std::decay_t<Expression>;
  using expression_traits =
      expression::detail::ExpressionTraits<expression_type>;

  constexpr static bool stores_references =
      expression_traits::stores_references || std::is_reference_v<Expression>;
  constexpr static bool is_const = std::is_const_v<Expression>;
  using value_type = typename expression_traits::value_type;
  using extents_type = typename expression_traits::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;

  auto expression() const & -> const Expression & { return m_expression; }
  auto expression() & -> Expression & { return m_expression; }
  auto expression() const && -> const Expression && {
    return std::move(m_expression);
  }
  auto expression() && -> Expression && { return std::move(m_expression); }
  auto extents() const -> extents_type { return expression().extents(); }
  [[nodiscard]] constexpr auto extent(rank_type i) const -> index_type {
    return m_expression.extent(i);
  }
  static constexpr auto static_extent(rank_type i) -> index_type {
    return Expression::static_extent(i);
  }
  // template <typename... Args>
  // ZipperBase(Args&&... v) : m_expression(std::forward<Args>(v)...) {}

  ZipperBase(expression_type &&v)
    requires(expression::concepts::OwningExpression<expression_type> &&
             !std::is_reference_v<Expression>)
      : m_expression(std::forward<Expression>(v)) {}
  ZipperBase(Derived &&v)
    requires(expression::concepts::OwningExpression<expression_type> &&
             !std::is_reference_v<Expression>)
      : ZipperBase(std::move(v.expression())) {}
  ZipperBase(ZipperBase &&v) = default;

  ZipperBase(const Derived &v)
    requires(expression::concepts::OwningExpression<expression_type> &&
             !std::is_reference_v<Expression>)
      : ZipperBase(v.expression()) {}
  ZipperBase(const expression_type &v)
    requires(expression::concepts::OwningExpression<expression_type> &&
             !std::is_reference_v<Expression>)
      : m_expression(v) {}
  ZipperBase(const ZipperBase &v) = default;
  auto operator=(const ZipperBase &) -> ZipperBase & = default;
  auto operator=(ZipperBase &&) -> ZipperBase & = default;

  /// In-place constructor: constructs the expression directly inside
  /// m_expression from forwarded arguments, avoiding any copy/move of the
  /// expression node.  This is essential for expressions that inherit
  /// NonReturnable (stores_references == true).
  template <typename... Args>
  ZipperBase(std::in_place_t, Args &&...args)
      : m_expression(std::forward<Args>(args)...) {}
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
    requires(!std::is_reference_v<Expression> &&
             expression::concepts::WritableExpression<expression_type> &&
             !is_const &&
             zipper::utils::extents::assignable_extents_v<
                 typename Other::extents_type, extents_type>)
      : m_expression(extents_traits::convert_from(other.extents())) {
    m_expression.assign(other);
  }

  // Removed: variadic forwarding constructor that silently enabled CTAD
  // from STL containers (std::vector, std::array) into MDSpan, which could
  // produce dangling references from rvalues.  All wrapper classes now have
  // their own variadic constructors that forward through std::in_place.

  // GCC's -Weffc++ warns about returning *this in assignment operators of
  // CRTP bases because derived() doesn't return the same type as *this.
  // A single suppression block covers all assignment/compound-assignment
  // operators that return derived().
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
  template <concepts::Expression Other>
  auto operator=(const Other &other) -> Derived &
    requires(expression::concepts::WritableExpression<expression_type> &&
             !is_const &&
             zipper::utils::extents::assignable_extents_v<
                 typename Other::extents_type, extents_type>)
  {
    m_expression.assign(other);
    return derived();
  }
  template <concepts::Expression Other>
  auto operator=(Other &&other) -> Derived &
    requires(expression::concepts::WritableExpression<expression_type> &&
             !is_const &&
             zipper::utils::extents::assignable_extents_v<
                 typename std::decay_t<Other>::extents_type, extents_type>)
  {
    m_expression.assign(other);
    return derived();
  }

  template <concepts::Zipper Other>
  auto operator+=(const Other &other) -> Derived &
    requires(expression::concepts::WritableExpression<expression_type> &&
             !is_const)
  {
    derived() = derived() + other;
    return derived();
  }
  template <concepts::Zipper Other>
  auto operator-=(const Other &other) -> Derived &
    requires(expression::concepts::WritableExpression<expression_type> &&
             !is_const)
  {
    derived() = derived() - other;
    return derived();
  }
  auto operator*=(const value_type &other) -> Derived &
    requires(expression::concepts::WritableExpression<expression_type> &&
             !is_const)
  {
    derived() = other * derived();
    return derived();
  }
  auto operator/=(const value_type &other) -> Derived &
    requires(expression::concepts::WritableExpression<expression_type> &&
             !is_const)
  {
    derived() = derived() / other;
    return derived();
  }
#pragma GCC diagnostic pop

  /// Returns a fully-owned copy of this expression that is safe to
  /// copy/move and can escape scope.  The returned wrapper has
  /// stores_references == false because all children store by value.
  /// Unlike eval(), which materializes everything to an MDArray,
  /// to_owned() preserves the lazy expression template structure but
  /// recursively deep-copies children so no references remain.
  auto to_owned() const {
    auto owned_expr = expression().make_owned();
    return DerivedT<const decltype(owned_expr)>(std::in_place,
                                                std::move(owned_expr));
  }

  /// Returns a wrapper that is copyable/movable/returnable even when
  /// this expression stores references.  The caller asserts that the
  /// referenced data will outlive all copies — hence "unsafe".
  /// Prefer to_owned() when a safe, owning copy is acceptable.
  ///
  /// Lvalue overloads store a reference to the expression; rvalue
  /// overloads move-own the expression node (the node itself may still
  /// hold internal references, e.g. Slice<MDArray&>).
  auto unsafe() const & {
    using V = expression::unary::UnsafeRef<const expression_type &>;
    return DerivedT<V>(std::in_place, expression());
  }
  auto unsafe() & {
    using V = expression::unary::UnsafeRef<expression_type &>;
    return DerivedT<V>(std::in_place, expression());
  }
  auto unsafe() && {
    using V = expression::unary::UnsafeRef<expression_type>;
    return DerivedT<V>(std::in_place, std::move(expression()));
  }

  template <typename OpType>
    requires(expression::unary::concepts::ScalarOperation<value_type, OpType>)
  auto unary_expr(const OpType &op) const {
    using V =
        expression::unary::CoefficientWiseOperation<const expression_type &,
                                                    OpType>;
    return DerivedT<V>(std::in_place, expression(), op);
  }

  template <template <typename> typename BaseType = DerivedT,
            rank_type... ranks>
  auto swizzle() const {
    using V = expression::unary::Swizzle<const expression_type &, ranks...>;
    return BaseType<V>(std::in_place, expression());
  }
  template <template <typename> typename BaseType = DerivedT,
            rank_type... ranks>
  auto swizzle() {
    using V = expression::unary::Swizzle<expression_type &, ranks...>;
    return BaseType<V>(std::in_place, expression());
  }
  template <typename T> auto cast() const {
    using V = expression::unary::Cast<T, const expression_type &>;
    return DerivedT<V>(std::in_place, expression());
  }

  auto diagonal() const {
    return VectorBase<expression::unary::Diagonal<const expression_type &>>(
        std::in_place, expression());
  }
  auto diagonal() {
    return VectorBase<expression::unary::Diagonal<expression_type &>>(
        std::in_place, expression());
  }

  template <concepts::IndexArgument... Args>
  auto operator()(Args &&...idxs) -> decltype(auto)
    requires(expression::concepts::WritableExpression<expression_type>)
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
    using V = expression::unary::Repeat<expression::unary::RepeatMode::Left,
                                        Count, const expression_type &>;
    return BaseType<V>(std::in_place, expression());
  }
  template <rank_type Count = 1,
            template <typename> typename BaseType = DerivedT>
  auto repeat_right() const {
    using V = expression::unary::Repeat<expression::unary::RepeatMode::Right,
                                        Count, const expression_type &>;
    return BaseType<V>(std::in_place, expression());
  }

protected:
  // slicing has fairly dimension specific effects for most derived types,
  // so we will just return the expression and let base class return things
  template <typename... Slices>
  auto slice_expression(Slices &&...slices) const {
    using my_expression_type =
        expression::unary::Slice<const expression_type &,
                                 detail::slice_type_for_t<std::decay_t<Slices>>...>;

    return my_expression_type(
        expression(),
        filter_args_for_zipperbase(std::forward<Slices>(slices))...);
  }

  template <typename... Slices> auto slice_expression() const {
    using my_expression_type =
        expression::unary::Slice<const expression_type &,
                                 detail::slice_type_for_t<std::decay_t<Slices>>...>;
    return my_expression_type(expression(), Slices{}...);
  }

  template <typename... Slices> auto slice_expression(Slices &&...slices) {
    using my_expression_type =
        expression::unary::Slice<expression_type &,
                                 detail::slice_type_for_t<std::decay_t<Slices>>...>;
    return my_expression_type(
        expression(),
        filter_args_for_zipperbase(std::forward<Slices>(slices))...);
  }
  template <typename... Slices> auto slice_expression() {
    using my_expression_type =
        expression::unary::Slice<expression_type &,
                                 detail::slice_type_for_t<std::decay_t<Slices>>...>;
    return my_expression_type(expression(), Slices{}...);
  }

public:
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
private:
  Expression m_expression;
};

} // namespace zipper

#endif
