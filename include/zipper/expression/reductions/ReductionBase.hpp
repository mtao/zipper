#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_REDUCTIONBASE_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_REDUCTIONBASE_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/ExpressionStorage.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"

namespace zipper::expression::reductions {

/// Common base class for all reduction operations.
///
/// Provides:
///   - Safe expression storage (by-reference for lvalues, by-value for rvalues)
///   - Common type aliases (expression_type, expression_traits, value_type)
///   - An expression() accessor
///
/// Template parameters:
///   Derived  — the CRTP derived reduction class
///   Expr     — the expression type, possibly ref-qualified (T& for lvalue,
///              T for rvalue). The storage mechanism uses expression_storage_t
///              to decide ref vs value storage.
///
/// Derived classes must remain `template <typename Expr> class Foo` (single
/// template param) to work as template-template parameters in PartialReduction.
template <typename Derived, typename Expr>
class ReductionBase {
public:
  using storage_type = zipper::detail::expression_storage_t<Expr>;
  using expression_type = std::remove_reference_t<Expr>;
  using expression_traits =
      zipper::expression::detail::ExpressionTraits<expression_type>;
  using value_type = typename expression_traits::value_type;

  /// Construct from a forwarding reference. Lvalues bind as references,
  /// rvalues are moved into owned storage.
  template <typename U>
    requires std::constructible_from<storage_type, U &&>
  explicit ReductionBase(U &&expr) : m_expression(std::forward<U>(expr)) {}

  ReductionBase(const ReductionBase &) = default;
  ReductionBase(ReductionBase &&) = default;
  auto operator=(const ReductionBase &) -> ReductionBase & = delete;
  auto operator=(ReductionBase &&) -> ReductionBase & = delete;

  auto expression() const -> const expression_type & { return m_expression; }

protected:
  storage_type m_expression;
};

} // namespace zipper::expression::reductions

#endif
