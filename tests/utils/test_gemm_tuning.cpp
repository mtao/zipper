// Per-TU kernel dimensioning: type-independent macros, plus a GemmTuning<T>
// specialization that overrides one type only. Must be its own TU (the
// configuration is fixed when zipper is included).

#define ZIPPER_GEMM_VECTOR_BYTES 16  // SSE-sized vectors, whatever -m flags say
#define ZIPPER_GEMM_SIMD_REGISTERS 16
#define ZIPPER_GEMM_TILE_NR 4

#include <zipper/expression/binary/detail/gemm_config.hpp>

namespace g = zipper::expression::binary::detail::gemm;

// float only: a deliberately different (narrow) tile.
template <>
struct g::GemmTuning<float> {
    static constexpr zipper::index_type vector_bytes = 16;
    static constexpr zipper::index_type simd_width = 4;
    static constexpr zipper::index_type registers = 16;
    static constexpr Tile tile{.mr = 4, .nr = 2};
};

#include <array>
#include <zipper/Matrix.hpp>

#include "../catch_include.hpp"

using namespace zipper;

TEST_CASE("gemm tuning: per-TU macros and GemmTuning specialization",
          "[gemm][tuning]") {
    // Macros apply to every type, in vector units: double = 2 lanes x 2
    // vectors x 4 columns.
    STATIC_CHECK(g::simd_width<double> == 2);
    STATIC_CHECK(g::default_tile<double>.mr == 4);
    STATIC_CHECK(g::default_tile<double>.nr == 4);
    // The specialization wins for float.
    STATIC_CHECK(g::simd_width<float> == 4);
    STATIC_CHECK(g::default_tile<float>.mr == 4);
    STATIC_CHECK(g::default_tile<float>.nr == 2);

    // Both still compute correctly.
    auto check = []<typename T>(T tol) {
        using M = Matrix<T, dynamic_extent, dynamic_extent>;
        for (auto [m, k, n] : {std::array<index_type, 3>{3, 5, 7},
                               std::array<index_type, 3>{41, 33, 29}}) {
            M A(m, k), B(k, n);
            for (index_type i = 0; i < m; ++i)
                for (index_type j = 0; j < k; ++j)
                    A(i, j) = T(1) / T(1 + i + 2 * j);
            for (index_type i = 0; i < k; ++i)
                for (index_type j = 0; j < n; ++j)
                    B(i, j) = T(1) / T(2 + 3 * i + j);
            M C = A * B;
            for (index_type i = 0; i < m; ++i)
                for (index_type j = 0; j < n; ++j) {
                    double ref = 0.0;
                    for (index_type p = 0; p < k; ++p)
                        ref += double(A(i, p)) * double(B(p, j));
                    CHECK(double(C(i, j))
                          == Catch::Approx(ref).epsilon(tol).margin(tol));
                }
        }
    };
    check(1e-12);
    check(1e-5f);
}
