#if !defined(ZIPPER_CONCEPTS_INNER_PRODUCT_SPACE_HPP)
#define ZIPPER_CONCEPTS_INNER_PRODUCT_SPACE_HPP

/// @file InnerProductSpace.hpp
/// @brief Richer algebraic structure concepts for vector-like types.
///
///   vector_space  ⊂  inner_product_space
///   vector_space  ⊂  normed_space
///
/// - inner_product_space:  vector space with a dot product.
/// - normed_space:         vector space with a norm.
///
/// Every inner product induces a norm (via Cauchy-Schwarz), but not
/// every norm comes from an inner product, so these are independent.

#include "Module.hpp"

#include <type_traits>

namespace zipper::concepts {

/// V is an inner product space: a vector space equipped with a dot
/// product that returns the scalar type.
template <typename V>
concept inner_product_space =
    vector_space<V>
    && requires(std::remove_cvref_t<V> a, std::remove_cvref_t<V> b) {
        { a.dot(b) } -> std::convertible_to<typename std::decay_t<V>::value_type>;
    };

/// V is a normed space: a vector space equipped with a norm that
/// returns the scalar type.
///
/// The norm operation typically requires sqrt to be available for the
/// scalar type (via ADL or std::sqrt).
template <typename V>
concept normed_space =
    vector_space<V>
    && requires(std::remove_cvref_t<V> v) {
        { v.norm() } -> std::convertible_to<typename std::decay_t<V>::value_type>;
    };

} // namespace zipper::concepts
#endif
