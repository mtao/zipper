#if !defined(ZIPPER_DETAIL_NONRETURNABLE_HPP)
#define ZIPPER_DETAIL_NONRETURNABLE_HPP

#include <type_traits>

namespace zipper::detail {

/// A mixin that prevents an expression from being copied or moved,
/// making it impossible to return from a function as an lvalue.
///
/// C++17 guaranteed copy elision still allows prvalue returns:
///   auto make() { return Foo(...); }   // OK — constructs in-place
/// but prevents:
///   auto make() { Foo f(...); return f; }   // ERROR — copy/move deleted
///
/// This is used to protect expression trees that store references from
/// escaping the scope of the objects they reference.
///
/// Expressions that need to be returned can use to_owned() to convert
/// the tree to a fully-owning form, or unsafe_ref() to opt out explicitly.
struct NonReturnable {
  NonReturnable() = default;
  NonReturnable(const NonReturnable &) = delete;
  NonReturnable(NonReturnable &&) = delete;
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
