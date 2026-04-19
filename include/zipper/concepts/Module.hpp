#if !defined(ZIPPER_CONCEPTS_MODULE_HPP)
#define ZIPPER_CONCEPTS_MODULE_HPP

/// @file Module.hpp
/// @brief Algebraic structure concepts for vector-like types.
///
/// Provides concepts for types that form modules and vector spaces:
///
///   module  ⊂  vector_space
///
/// - module:        additive abelian group with scalar multiplication
///                  by a ring.  E.g., Vector<int, 3>.
/// - vector_space:  module whose scalars form a field.
///                  E.g., Vector<double, 3>.
///
/// Both concepts have a binary form (explicit scalar ring/field type)
/// and a unary convenience form that infers the scalar from
/// V::value_type.

#include "Algebraic.hpp"

#include <type_traits>

namespace zipper::concepts {

/// V is a module over ring R: V supports addition, subtraction,
/// negation, and scalar multiplication by R.
///
/// No return-type checking on operations because expression-template
/// systems (like zipper's own) return proxy types rather than V itself.
template <typename V, typename R>
concept module_over =
    ring<R>
    && requires(std::remove_cvref_t<V> a, std::remove_cvref_t<V> b, R s) {
        a + b;
        a - b;
        -a;
        s * a;
    };

/// V is a module over its own value_type.
template <typename V>
concept module =
    requires { typename std::decay_t<V>::value_type; }
    && module_over<V, typename std::decay_t<V>::value_type>;

/// V is a vector space over field F: a module where the scalar type
/// supports meaningful division.
template <typename V, typename F>
concept vector_space_over = module_over<V, F> && field<F>;

/// V is a vector space over its own value_type.
template <typename V>
concept vector_space =
    requires { typename std::decay_t<V>::value_type; }
    && vector_space_over<V, typename std::decay_t<V>::value_type>;

} // namespace zipper::concepts
#endif
