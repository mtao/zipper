#if !defined(ZIPPER_STATIC_SCALAR_HPP)
#define ZIPPER_STATIC_SCALAR_HPP

/// @file static_scalar.hpp
/// @brief Compile-time scalars: std::integral_constant, plus the concepts and
/// helpers zipper uses to recognize them.
///
/// A scalar whose value is part of its type is a
/// `std::integral_constant<T, v>` — since C++20 any type usable as a template
/// argument works, including floating point (`std::integral_constant<double,
/// 0.5>`). `zipper::cw<v>` is shorthand for one, with T deduced from the
/// literal (`cw<1.0>` is a double, `cw<1>` an int); it mirrors C++26
/// `std::cw` / `std::constant_wrapper`.
///
/// Static scalars are used as coefficients in BLAS-style interfaces
/// (`gemm(alpha, A, B, beta, C)` takes run-time or static alpha / beta;
/// static 0 / 1 / -1 pick cheaper code paths).
///
/// Note that a floating-point template argument must be written as a
/// floating-point literal: `std::integral_constant<double, 1.0>`, not `1`.

#include <concepts>
#include <type_traits>

#include "zipper/detail/is_integral_constant.hpp"

namespace zipper {

/// `std::integral_constant<decltype(V), V>{}` (cf. C++26 `std::cw`).
template <auto V>
inline constexpr std::integral_constant<decltype(V), V> cw{};

namespace concepts {
    /// S is a compile-time scalar (a std::integral_constant).
    template <typename S>
    concept StaticScalar =
        zipper::detail::is_integral_constant_v<std::remove_cvref_t<S>>;

    /// S is a compile-time scalar equal to V.
    template <typename S, auto V>
    concept StaticScalarOf =
        StaticScalar<S> && (std::remove_cvref_t<S>::value == V);

    /// S is a coefficient for values of type T — what scales a T-valued
    /// expression in BLAS-style interfaces (alpha / beta): a run-time scalar
    /// convertible to T, or a compile-time one (static 0 / 1 / -1 select
    /// cheaper code paths).
    ///
    ///   template <concepts::Coefficient<value_type> Alpha>
    ///   void accumulate(..., Alpha alpha);
    template <typename S, typename T>
    concept Coefficient =
        (StaticScalar<S>
         && std::convertible_to<typename std::remove_cvref_t<S>::value_type, T>)
        || std::convertible_to<S, T>;
} // namespace concepts

/// The product of two T coefficients: compile-time when both are (and T can
/// be a template argument), otherwise a T. Multiplying by a static 1 is free
/// (returns the other operand unchanged).
template <typename T, typename A, typename B>
constexpr auto scalar_product(A a, B b) {
    if constexpr (concepts::StaticScalarOf<A, 1>) {
        return b;
    } else if constexpr (concepts::StaticScalarOf<B, 1>) {
        return a;
    } else if constexpr (concepts::StaticScalar<A> && concepts::StaticScalar<B>
                         && std::is_arithmetic_v<T>) {
        return std::integral_constant<T, static_cast<T>(A::value)
                                             * static_cast<T>(B::value)>{};
    } else {
        return static_cast<T>(a) * static_cast<T>(b);
    }
}

} // namespace zipper

#endif
