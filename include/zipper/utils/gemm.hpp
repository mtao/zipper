#if !defined(ZIPPER_UTILS_GEMM_HPP)
#define ZIPPER_UTILS_GEMM_HPP

/// @file gemm.hpp
/// @brief BLAS-style general matrix multiply: C = beta * C + alpha * A * B.
/// @ingroup linear_algebra
///
/// `zipper::utils::gemm(alpha, A, B, beta, C)` is the explicit form of what
/// the operators do implicitly (`C = A * B`, `C.noalias() += A * B`,
/// `C = 2 * (A * B)`, ...). alpha and beta are run-time scalars or
/// `zipper::cw<V>`:
///
///   - beta = 0 never reads C (garbage / NaN in C does not leak through);
///   - beta = 1 accumulates; alpha = 1 skips the scaling;
///   - run-time 0 / 1 are detected once per call and take the same paths.
///
/// Dense dynamic floating-point operands go through the blocked GEMM kernel;
/// anything else (e.g. small static matrices) uses the generic product.
///
/// Like BLAS, C must not alias A or B (checked cheaply in debug builds for the
/// kernel path). Use the operators when aliasing is possible.
///
/// @code
///   using zipper::cw;
///   zipper::utils::gemm(cw<1.0>, A, B, cw<0.0>, C);  // C = A B
///   zipper::utils::gemm(-h, A, B, cw<1.0>, Y);       // Y -= h A B
///   zipper::utils::gemm(alpha, A, B, beta, C);                        // run time
///   zipper::utils::gemm(std::execution::par, alpha, A, B, beta, C);  // parallel
/// @endcode

#include <execution>
#include <type_traits>
#include <utility>

#include "zipper/concepts/Matrix.hpp"
#include "zipper/static_scalar.hpp"

namespace zipper::utils {

/// C = beta * C + alpha * A * B under an execution policy
/// (std::execution::par parallelizes the kernel over row blocks of C; with
/// libstdc++ this needs TBB linked). The policy only affects the kernel path.
template <typename Policy,
          typename Alpha,
          zipper::concepts::Matrix AType,
          zipper::concepts::Matrix BType,
          typename Beta,
          zipper::concepts::Matrix CType>
    requires std::is_execution_policy_v<std::remove_cvref_t<Policy>>
             && zipper::concepts::Coefficient<
                 Alpha, typename std::remove_cvref_t<CType>::value_type>
             && zipper::concepts::Coefficient<
                 Beta, typename std::remove_cvref_t<CType>::value_type>
auto gemm(const Policy &policy,
          Alpha alpha,
          const AType &A,
          const BType &B,
          Beta beta,
          CType &&C) -> CType && {
    using T = typename std::remove_cvref_t<CType>::value_type;
    const auto product = A * B;
    auto &to = C.expression();
    if constexpr (requires {
                      product.expression().accumulate_to(
                          to, alpha, beta, policy);
                  }) {
        product.expression().accumulate_to(to, alpha, beta, policy);
    } else {
        // Generic fallback with the same alpha/beta semantics.
        const auto P = product.eval();
        if constexpr (zipper::concepts::StaticScalarOf<Alpha, 0>) {
            if constexpr (zipper::concepts::StaticScalarOf<Beta, 0>) {
                C.set_zero();
            } else if constexpr (!zipper::concepts::StaticScalarOf<Beta, 1>) {
                C *= static_cast<T>(beta);
            }
        } else if constexpr (zipper::concepts::StaticScalarOf<Beta, 0>) {
            C = static_cast<T>(alpha) * P;
        } else if constexpr (!zipper::concepts::StaticScalar<Beta>) {
            if (static_cast<T>(beta) == T(0)) {
                C = static_cast<T>(alpha) * P; // never read C
            } else {
                C = static_cast<T>(beta) * C + static_cast<T>(alpha) * P;
            }
        } else {
            C = static_cast<T>(beta) * C + static_cast<T>(alpha) * P;
        }
    }
    return std::forward<CType>(C);
}

/// C = beta * C + alpha * A * B, sequential. See the file documentation.
template <typename Alpha,
          zipper::concepts::Matrix AType,
          zipper::concepts::Matrix BType,
          typename Beta,
          zipper::concepts::Matrix CType>
    requires zipper::concepts::Coefficient<
                 Alpha, typename std::remove_cvref_t<CType>::value_type>
             && zipper::concepts::Coefficient<
                 Beta, typename std::remove_cvref_t<CType>::value_type>
auto gemm(Alpha alpha, const AType &A, const BType &B, Beta beta, CType &&C)
    -> CType && {
    return gemm(std::execution::seq, alpha, A, B, beta, std::forward<CType>(C));
}

} // namespace zipper::utils

#endif
