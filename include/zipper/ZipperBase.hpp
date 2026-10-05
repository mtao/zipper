#if !defined(ZIPPER_ZIPPERBASE_HPP)
#define ZIPPER_ZIPPERBASE_HPP

#include "concepts/Expression.hpp"
#include "concepts/IndexArgument.hpp"
#include "concepts/Zipper.hpp"
#include "detail/NoAliasProxy.hpp"
#include "detail/NonReturnable.hpp"
#include "expression/concepts/capabilities.hpp"
#include "expression/detail/AssignHelper.hpp"
#include "expression/nullary/Zero.hpp"
#include "static_scalar.hpp"
#include "expression/unary/Cast.hpp"
#include "expression/unary/CoefficientWiseOperation.hpp"
#include "expression/unary/DiagonalExtract.hpp"
#include "expression/unary/Lift.hpp"
#include "expression/unary/Repeat.hpp"
#include "expression/unary/Slice.hpp"
#include "expression/unary/Swizzle.hpp"
#include "expression/unary/UnsafeRef.hpp"
#include "expression/unary/concepts/ScalarOperation.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/detail/member_child_storage.hpp"
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
} // namespace detail

template <template <typename> typename DerivedT,
          concepts::QualifiedExpression Expression>
class ZipperBase
  : public detail::returnability_mixin_t<
        expression::detail::ExpressionTraits<
            std::decay_t<Expression>>::stores_references
        || std::is_reference_v<Expression>> {
  public:
    ZipperBase()
        requires(std::is_default_constructible_v<Expression>)
      : m_expression() {}
    using Derived = DerivedT<Expression>;
    auto derived(this auto &self) -> auto & {
        if constexpr (std::is_const_v<
                          std::remove_reference_t<decltype(self)>>) {
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
    constexpr static bool is_const =
        std::is_const_v<std::remove_reference_t<Expression>>;
    using value_type = typename expression_traits::value_type;

  private:
    template <typename Other, concepts::Coefficient<value_type> Alpha>
    void accumulate(const Other &other, Alpha alpha) {
        expression::detail::AssignHelper<
            typename Other::expression_type,
            expression_type>::accumulate(other.expression(), m_expression, alpha);
    }

  public:
    using extents_type = typename expression_traits::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;

    auto expression() const & -> const expression_type & {
        return m_expression;
    }
    auto expression() & -> Expression & { return m_expression; }
    auto expression()
        const && -> std::conditional_t<std::is_reference_v<Expression>,
                                       const expression_type &,
                                       const expression_type &&> {
        if constexpr (std::is_reference_v<Expression>) {
            return m_expression;
        } else {
            return std::move(m_expression);
        }
    }
    auto expression()
        && -> std::conditional_t<std::is_reference_v<Expression>,
                                 std::remove_reference_t<Expression> &,
                                 Expression &&> {
        if constexpr (std::is_reference_v<Expression>) {
            return m_expression;
        } else {
            return std::move(m_expression);
        }
    }
    auto extents() const -> extents_type { return expression().extents(); }
    [[nodiscard]] constexpr auto extent(rank_type i) const -> index_type {
        return m_expression.extent(i);
    }
    static constexpr auto static_extent(rank_type i) -> index_type {
        return expression_type::static_extent(i);
    }
    // template <typename... Args>
    // ZipperBase(Args&&... v) : m_expression(std::forward<Args>(v)...) {}

    ZipperBase(expression_type &&v)
        requires(expression::concepts::OwningExpression<expression_type>
                 && !std::is_reference_v<Expression>)
      : m_expression(std::forward<Expression>(v)) {}
    ZipperBase(Derived &&v)
        requires(expression::concepts::OwningExpression<expression_type>
                 && !std::is_reference_v<Expression>)
      : ZipperBase(std::move(v.expression())) {}
    ZipperBase(ZipperBase &&v) = default;

    ZipperBase(const Derived &v)
        requires(expression::concepts::OwningExpression<expression_type>
                 && !std::is_reference_v<Expression>)
      : ZipperBase(v.expression()) {}
    ZipperBase(const expression_type &v)
        requires(expression::concepts::OwningExpression<expression_type>
                 && !std::is_reference_v<Expression>)
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

    /// Uninitialized constructor: forwards zipper::uninitialized (plus any
    /// extents arguments) to the owned expression, skipping zero-fill.
    template <typename... Args>
        requires(std::is_constructible_v<Expression, uninitialized_t, Args...>)
    explicit ZipperBase(uninitialized_t, Args &&...args)
      : m_expression(uninitialized, std::forward<Args>(args)...) {}
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
        requires(!std::is_reference_v<Expression>
                 && expression::concepts::WritableExpression<expression_type>
                 && !is_const
                 && zipper::utils::extents::assignable_extents_v<
                     typename Other::extents_type,
                     extents_type>)
      : m_expression(make_destination(
            extents_traits::convert_from(other.extents()))) {
        if constexpr (detail::is_fresh_mdarray_v<expression_type>) {
            // Fresh owning storage is already shaped and cannot alias the
            // source. evaluate_to writes every coefficient, so the storage
            // was allocated without a zero-fill (see make_destination).
            expression::detail::AssignHelper<Other, expression_type>::
                evaluate_to(other, m_expression);
        } else {
            m_expression.assign(other);
        }
    }

    /// Explicitly promise that RHS reads do not alias this destination's
    /// storage. Available for types that accept the NoAliasDestination concept.
    /// Rank-zero RHS preserves the shape.
    template <typename Self>
    auto noalias(this Self &self) -> detail::NoAliasProxy<Self>
        requires detail::NoAliasDestination<Self>
    {
        return detail::NoAliasProxy<Self>(self);
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
        requires(expression::concepts::WritableExpression<expression_type>
                 && !is_const
                 && zipper::utils::extents::assignable_extents_v<
                     typename Other::extents_type,
                     extents_type>)
    {
        m_expression.assign(other);
        return derived();
    }
    template <concepts::Expression Other>
    auto operator=(Other &&other) -> Derived &
        requires(expression::concepts::WritableExpression<expression_type>
                 && !is_const
                 && zipper::utils::extents::assignable_extents_v<
                     typename std::decay_t<Other>::extents_type,
                     extents_type>)
    {
        m_expression.assign(other);
        return derived();
    }

    /// Assigns from a Zipper-wrapped expression by unwrapping to the
    /// expression layer.  This enables compound assignment operators
    /// (`+=`, `-=`, etc.) whose RHS is a Zipper wrapper rather than a
    /// raw Expression.
    template <concepts::Zipper Other>
    auto operator=(const Other &other) -> Derived &
        requires(expression::concepts::WritableExpression<expression_type>
                 && !is_const
                 && zipper::utils::extents::assignable_extents_v<
                     typename std::decay_t<Other>::extents_type,
                     extents_type>)
    {
        m_expression.assign(other.expression());
        return derived();
    }

    /// `x += y` / `x -= y` (requires `x + y` / `x - y` to be valid, so the
    /// usual semantic rules apply). AssignHelper::accumulate does the update:
    /// in place when provably alias-free, otherwise from a snapshot of y;
    /// accumulable y (e.g. a matrix product) are evaluated through their fast
    /// path. `x.noalias() += y` skips the aliasing check.
    template <concepts::Zipper Other>
    auto operator+=(const Other &other) -> Derived &
        requires(expression::concepts::WritableExpression<expression_type>
                 && !is_const
                 && requires(const Derived &x, const Other &y) { x + y; })
    {
        accumulate(other, zipper::cw<1>);
        return derived();
    }
    template <concepts::Zipper Other>
    auto operator-=(const Other &other) -> Derived &
        requires(expression::concepts::WritableExpression<expression_type>
                 && !is_const
                 && requires(const Derived &x, const Other &y) { x - y; })
    {
        accumulate(other, zipper::cw<-1>);
        return derived();
    }
    auto operator*=(const value_type &other) -> Derived &
        requires(expression::concepts::WritableExpression<expression_type>
                 && !is_const)
    {
        derived() = other * derived();
        return derived();
    }
    auto operator/=(const value_type &other) -> Derived &
        requires(expression::concepts::WritableExpression<expression_type>
                 && !is_const)
    {
        derived() = derived() / other;
        return derived();
    }

    /// Set every coefficient to zero (value_type{}). Assigns a structural
    /// `Zero` of this object's extents, so zero-aware expressions and sparse
    /// storage see the zero (sparse storage is simply cleared).
    auto set_zero() -> Derived &
        requires(expression::concepts::WritableExpression<expression_type>
                 && !is_const)
    {
        [&]<index_type... Is>(const zipper::extents<Is...> &e) {
            m_expression.assign(
                expression::nullary::Zero<value_type, Is...>(e));
        }(extents());
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
    /// overloads own value-stored expression nodes and preserve
    /// reference-stored referents (nodes may also hold internal references,
    /// e.g. Slice<MDArray&>).
    auto unsafe() const & {
        using V = expression::unary::UnsafeRef<const expression_type &>;
        return DerivedT<V>(std::in_place, expression());
    }
    auto unsafe() & {
        using V =
            expression::unary::UnsafeRef<std::remove_reference_t<Expression> &>;
        return DerivedT<V>(std::in_place, expression());
    }
    auto unsafe() && {
        using child_t = detail::member_child_storage_t<Derived, Expression>;
        using V = expression::unary::UnsafeRef<child_t>;
        return DerivedT<V>(std::in_place, std::move(*this).expression());
    }
    auto unsafe() const && {
        using child_t =
            detail::member_child_storage_t<const Derived, Expression>;
        using V = expression::unary::UnsafeRef<child_t>;
        return DerivedT<V>(std::in_place, std::move(*this).expression());
    }

    /// Returns a view-propagating wrapper.  Like unsafe(), the caller asserts
    /// that the referenced data outlives all copies.  Unlike unsafe(), views
    /// derived from a ref() result (head, tail, row, transpose, etc.) are
    /// also Returnable — the view-propagating flag causes child storage to
    /// copy the lightweight UnsafeRef by value instead of by reference.
    ///
    /// Usage:
    ///   auto r = x.ref();       // r is Returnable
    ///   auto h = r.head<3>();   // h is also Returnable (unlike x.unsafe())
    auto ref() const & -> DerivedT<
        expression::unary::UnsafeRef<const expression_type &, true>> {
        using V = expression::unary::UnsafeRef<const expression_type &, true>;
        return DerivedT<V>(std::in_place, expression());
    }
    auto ref() & -> DerivedT<
        expression::unary::UnsafeRef<std::remove_reference_t<Expression> &,
                                     true>> {
        using V =
            expression::unary::UnsafeRef<std::remove_reference_t<Expression> &,
                                         true>;
        return DerivedT<V>(std::in_place, expression());
    }
    auto ref() && = delete; // rvalue ref() is nonsensical — no lvalue to bind
    auto ref() const && = delete;

    template <typename OpType, typename Self>
        requires(
            expression::unary::concepts::ScalarOperation<value_type, OpType>
            && std::constructible_from<OpType, const OpType &>)
    auto unary_expr(this Self &&self, const OpType &op) {
        using child_t =
            detail::member_child_storage_t<Self, raw_expression_type>;
        using V = expression::unary::CoefficientWiseOperation<child_t, OpType>;
        return DerivedT<V>(
            std::in_place, std::forward<Self>(self).expression(), op);
    }

    template <template <typename> typename BaseType = DerivedT,
              rank_type... ranks,
              typename Self>
    auto swizzle(this Self &&self) {
        using child_t =
            detail::member_child_storage_t<Self, raw_expression_type>;
        using V = expression::unary::Swizzle<child_t, ranks...>;
        return BaseType<V>(std::in_place,
                           std::forward<Self>(self).expression());
    }
    template <typename T, typename Self>
    auto cast(this Self &&self) {
        using child_t =
            detail::member_child_storage_t<Self, raw_expression_type>;
        using V = expression::unary::Cast<T, child_t>;
        return DerivedT<V>(std::in_place,
                           std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto diagonal(this Self &&self) {
        using child_t =
            detail::member_child_storage_t<Self, raw_expression_type>;
        return VectorBase<expression::unary::DiagonalExtract<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <concepts::IndexArgument... Args>
    auto operator()(Args &&...idxs) -> decltype(auto)
        requires(expression::concepts::WritableExpression<expression_type>
                 && !is_const)
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

    // appends Count dummy dimensions after the existing extents
    template <rank_type Count = 1,
              template <typename> typename BaseType = DerivedT,
              typename Self>
    auto lift(this Self &&self) {
        using child_t =
            detail::member_child_storage_t<Self, raw_expression_type>;
        using V = expression::unary::Lift<Count, child_t>;
        return BaseType<V>(std::in_place,
                           std::forward<Self>(self).expression());
    }

    // deprecated: use lift() instead
    template <rank_type Count = 1,
              template <typename> typename BaseType = DerivedT,
              typename Self>
    auto repeat_left(this Self &&self) {
        using child_t =
            detail::member_child_storage_t<Self, raw_expression_type>;
        using V = expression::unary::
            Repeat<expression::unary::RepeatMode::Left, Count, child_t>;
        return BaseType<V>(std::in_place,
                           std::forward<Self>(self).expression());
    }
    // deprecated: use lift() instead
    template <rank_type Count = 1,
              template <typename> typename BaseType = DerivedT,
              typename Self>
    auto repeat_right(this Self &&self) {
        using child_t =
            detail::member_child_storage_t<Self, raw_expression_type>;
        using V = expression::unary::
            Repeat<expression::unary::RepeatMode::Right, Count, child_t>;
        return BaseType<V>(std::in_place,
                           std::forward<Self>(self).expression());
    }

  protected:
    // slicing has fairly dimension specific effects for most derived types,
    // so we will just return the expression and let base class return things
    template <typename... Slices, typename Self>
    auto slice_expression(this Self &&self, Slices &&...slices) {
        using child_t =
            detail::member_child_storage_t<Self, raw_expression_type>;
        using my_expression_type = expression::unary::
            Slice<child_t, detail::slice_type_for_t<std::decay_t<Slices>>...>;

        return my_expression_type(
            std::forward<Self>(self).expression(),
            filter_args_for_zipperbase(std::forward<Slices>(slices))...);
    }

    template <typename... Slices, typename Self>
    auto slice_expression(this Self &&self) {
        using child_t =
            detail::member_child_storage_t<Self, raw_expression_type>;
        using my_expression_type = expression::unary::
            Slice<child_t, detail::slice_type_for_t<std::decay_t<Slices>>...>;
        return my_expression_type(std::forward<Self>(self).expression(),
                                  Slices{}...);
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
    /// Builds the destination for the converting constructor. Fresh
    /// MDArray storage is fully overwritten by evaluate_to, so it skips the
    /// zero-fill; any other expression type is constructed normally.
    static auto make_destination(const extents_type &e) -> expression_type {
        if constexpr (detail::is_fresh_mdarray_v<expression_type>) {
            return expression_type(zipper::uninitialized, e);
        } else {
            return expression_type(e);
        }
    }

    Expression m_expression;
};

} // namespace zipper

#endif
