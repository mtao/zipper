#if !defined(ZIPPER_DETAIL_NONRETURNABLE_HPP)
#define ZIPPER_DETAIL_NONRETURNABLE_HPP

#include <type_traits>

namespace zipper::detail {

/// A mixin that prevents an expression from being **copied** while still
/// allowing moves.  This protects expression trees that store references
/// from being accidentally duplicated (e.g. `auto x = lvalue_expr;`).
///
/// Moves are allowed so that operator chaining works:
///   auto chain = (a + b) * 2.0;   // OK — intermediate moved into outer op
///
/// Returning a named local is technically possible via implicit move,
/// which is a deliberate trade-off: compile-time prevention of ALL
/// escaping (deleting both copy and move) would also prevent the
/// essential operator-chaining use case.  Use `stores_references` for
/// compile-time checks in function signatures, `to_owned()` / `eval()`
/// to create safe copies, or `unsafe_ref()` to explicitly opt out.
struct NonReturnable {
  NonReturnable() = default;
  NonReturnable(const NonReturnable &) = delete;
  NonReturnable(NonReturnable &&) = default;
  auto operator=(const NonReturnable &) -> NonReturnable & = delete;
  auto operator=(NonReturnable &&) -> NonReturnable & = delete;

protected:
  ~NonReturnable() = default;
};

/// Empty base for expressions that don't store references (no restrictions).
struct Returnable {
  Returnable() = default;
  Returnable(const Returnable &) = default;
  Returnable(Returnable &&) = default;
  auto operator=(const Returnable &) -> Returnable & = default;
  auto operator=(Returnable &&) -> Returnable & = default;

protected:
  ~Returnable() = default;
};

/// Select the appropriate mixin based on whether the expression stores
/// references.
template <bool StoresReferences>
using returnability_mixin_t =
    std::conditional_t<StoresReferences, NonReturnable, Returnable>;

} // namespace zipper::detail

#endif
