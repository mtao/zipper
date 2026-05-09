/// @file general.hpp
/// @brief General eigenvalue decomposition for non-symmetric matrices.
/// @ingroup eigenvalue
///
/// Provides a unified entry point `eigen()` that dispatches to the
/// appropriate algorithm:
///   - For symmetric matrices: `symmetric_eigen()` (tridiagonal QR)
///   - For general matrices: `hessenberg_qr_eigen()` (Schur decomposition)
///
/// Also re-exports the specific types and functions from the sub-modules
/// for direct access.

#if !defined(ZIPPER_UTILS_EIGENVALUE_GENERAL_HPP)
#define ZIPPER_UTILS_EIGENVALUE_GENERAL_HPP

#include <zipper/utils/eigenvalue/hessenberg_qr.hpp>
#include <zipper/utils/eigenvalue/symmetric.hpp>
#include <zipper/utils/eigenvalue/tridiagonal_qr.hpp>

namespace zipper::utils::eigenvalue {

// All types and functions are already available through the individual headers:
//   - SymmetricEigenResult, symmetric_eigen()      from symmetric.hpp
//   - EigenResult, hessenberg_qr_eigen()            from hessenberg_qr.hpp
//   - TridiagonalEigenResult, tridiagonal_qr_eigen() from tridiagonal_qr.hpp

} // namespace zipper::utils::eigenvalue

#endif
