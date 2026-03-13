#if !defined(ZIPPER_EXPRESSION_BINARY_BINARYEXPRESSIONBASE_HPP)
#define ZIPPER_EXPRESSION_BINARY_BINARYEXPRESSIONBASE_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/ExpressionStorage.hpp"
#include "zipper/expression/ExpressionBase.hpp"

namespace zipper::expression::binary {

namespace detail {

/// Default traits for binary expressions. Inherits from BasicExpressionTraits
/// to provide the access_features/shape_features interface.
///
/// ChildA / ChildB are the *qualified* child types as passed to the binary
/// expression class template (e.g. `const MDArray<…>&` for reference
/// storage, `const MDArray<…>` for owning storage).  Traits are looked
/// up on the decayed (unqualified) type.
template <zipper::concepts::QualifiedExpression ChildA,
          zipper::concepts::QualifiedExpression ChildB>
struct DefaultBinaryExpressionTraits
    : public expression::detail::BasicExpressionTraits<
          typename expression::detail::ExpressionTraits<std::decay_t<ChildA>>::value_type,
          zipper::dextents<0>,
          expression::detail::AccessFeatures{
              .is_const = true, .is_reference = false, .is_alias_free = true},
          expression::detail::ShapeFeatures{.is_resizable = false}> {
  using ATraits = expression::detail::ExpressionTraits<std::decay_t<ChildA>>;
  using BTraits = expression::detail::ExpressionTraits<std::decay_t<ChildB>>;
  using lhs_value_type = typename ATraits::value_type;
  using rhs_value_type = typename BTraits::value_type;
  static_assert(std::is_convertible_v<typename ATraits::value_type,
                                      typename BTraits::value_type> ||
                std::is_convertible_v<typename BTraits::value_type,
                                      typename ATraits::value_type>);

  // defaulting to first parameter
  using value_type = typename ATraits::value_type;
  constexpr static bool is_coefficient_consistent =
      ATraits::is_coefficient_consistent && BTraits::is_coefficient_consistent;
  constexpr static bool is_value_based = true;

  /// stores_references is true when either child is stored by reference,
  /// OR when either child (even if stored by value) internally stores
  /// references to external data.  This ensures that an expression like
  /// (a+b)+(c+d) correctly reports stores_references==true when the sub-
  /// expressions are moved in by value but still hold references to a,b,c,d.
  constexpr static bool stores_references =
      std::is_reference_v<ChildA> || std::is_reference_v<ChildB> ||
      ATraits::stores_references || BTraits::stores_references;
};
} // namespace detail

/// Base class for binary expression nodes.
///
/// ChildTypeA / ChildTypeB should be const-ref-qualified (e.g. `const A&`)
/// when the caller wants reference storage (the common case — children outlive
/// the expression node).  Pass a non-reference type to trigger by-value
/// storage (e.g. for temporaries that must be owned).
///
/// The storage mechanism is provided by expression_storage_t:
///   - `const A&`  →  stores as `const A&`  (reference)
///   - `const A`   →  stores as `const A`   (by value, owns it)
template <typename Derived, typename ChildTypeA, typename ChildTypeB>
class BinaryExpressionBase
    : public expression::ExpressionBase<Derived> {
public:
  using lhs_storage_type = zipper::detail::expression_storage_t<ChildTypeA>;
  using rhs_storage_type = zipper::detail::expression_storage_t<ChildTypeB>;
  using lhs_expression_type = std::remove_reference_t<ChildTypeA>;
  using rhs_expression_type = std::remove_reference_t<ChildTypeB>;

  using self_type = BinaryExpressionBase<Derived, ChildTypeA, ChildTypeB>;
  using traits = zipper::expression::detail::ExpressionTraits<Derived>;
  using extents_type = typename traits::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  using value_type = typename traits::value_type;
  constexpr static bool is_static = extents_traits::is_static;
  constexpr static bool is_value_based = traits::is_value_based;

  using Base = expression::ExpressionBase<Derived>;

  auto derived(this auto& self) -> auto& {
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(self)>>) {
      return static_cast<const Derived &>(self);
    } else {
      return static_cast<Derived &>(self);
    }
  }
  using lhs_value_type = typename traits::lhs_value_type;
  using rhs_value_type = typename traits::rhs_value_type;

  BinaryExpressionBase(const BinaryExpressionBase &) = default;
  BinaryExpressionBase(BinaryExpressionBase &&) = default;
  auto operator=(const BinaryExpressionBase &)
      -> BinaryExpressionBase & = delete;
  auto operator=(BinaryExpressionBase &&) -> BinaryExpressionBase & = delete;

  template <typename U, typename V>
    requires std::constructible_from<lhs_storage_type, U &&> &&
             std::constructible_from<rhs_storage_type, V &&>
  BinaryExpressionBase(U &&a, V &&b)
      : m_lhs(std::forward<U>(a)), m_rhs(std::forward<V>(b)) {}

  /// Default extents() constructs from extent() calls.
  /// Derived classes must provide extent(rank_type) — there is no single
  /// child to delegate to.
  constexpr auto extents() const -> extents_type {
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    return extents_traits::make_extents_from(derived());
  }

  auto lhs() const -> const lhs_expression_type & { return m_lhs; }

  auto rhs() const -> const rhs_expression_type & { return m_rhs; }

  auto get_value(const lhs_value_type &l, const rhs_value_type &r) const
      -> value_type {
    return derived().get_value(l, r);
  }

  template <typename... Args>
  auto coeff(Args &&...args) const -> value_type
    requires(is_value_based)
  {
    return value_type(get_value(m_lhs(std::forward<Args>(args)...),
                                m_rhs(std::forward<Args>(args)...)));
  }

private:
  lhs_storage_type m_lhs;
  rhs_storage_type m_rhs;
};

} // namespace zipper::expression::binary
#endif
