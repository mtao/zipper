
#include "../catch_include.hpp"

#include <zipper/Array.hpp>
#include <zipper/DataArray.hpp>
#include <zipper/Form.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Quaternion.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>

#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/nullary/MDSpan.hpp>
#include <zipper/expression/nullary/Unit.hpp>

#include <zipper/storage/DynamicDenseData.hpp>
#include <zipper/storage/SpanData.hpp>
#include <zipper/storage/StaticDenseData.hpp>

// ═══════════════════════════════════════════════════════════════════════════
// Layer 1: Leaf Storage Types
//
// These are raw data containers.  They should be exactly the size of their
// underlying std container — no extra members, no vtable, no overhead.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sizeof StaticDenseData", "[sizeof][storage]") {
  using zipper::storage::StaticDenseData;

  // StaticDenseData<T,N> wraps std::array<T,N> and nothing else.
  STATIC_REQUIRE(sizeof(StaticDenseData<float, 1>) == sizeof(float) * 1);
  STATIC_REQUIRE(sizeof(StaticDenseData<float, 3>) == sizeof(float) * 3);
  STATIC_REQUIRE(sizeof(StaticDenseData<float, 4>) == sizeof(float) * 4);
  STATIC_REQUIRE(sizeof(StaticDenseData<double, 3>) == sizeof(double) * 3);
  STATIC_REQUIRE(sizeof(StaticDenseData<double, 9>) == sizeof(double) * 9);
  STATIC_REQUIRE(sizeof(StaticDenseData<double, 16>) == sizeof(double) * 16);

  // Edge: N=0 clamps to 1 element (see StaticDenseData implementation)
  STATIC_REQUIRE(sizeof(StaticDenseData<double, 0>) == sizeof(double) * 1);
}

TEST_CASE("sizeof SpanData", "[sizeof][storage]") {
  using zipper::storage::SpanData;

  // SpanData<T,N> inherits std::span<T,N> — should be same size.
  STATIC_REQUIRE(sizeof(SpanData<float, 3>) == sizeof(std::span<float, 3>));
  STATIC_REQUIRE(sizeof(SpanData<double, 3>) == sizeof(std::span<double, 3>));
  STATIC_REQUIRE(sizeof(SpanData<const double, 3>) ==
                 sizeof(std::span<const double, 3>));

  // Static span is just a pointer (no size stored).
  STATIC_REQUIRE(sizeof(SpanData<float, 3>) == sizeof(float *));

  // Dynamic span stores pointer + size.
  STATIC_REQUIRE(sizeof(SpanData<float, std::dynamic_extent>) ==
                 sizeof(std::span<float, std::dynamic_extent>));
}

TEST_CASE("sizeof DynamicDenseData", "[sizeof][storage]") {
  using zipper::storage::DynamicDenseData;

  // DynamicDenseData<T> wraps std::vector<T> — should be same size.
  STATIC_REQUIRE(sizeof(DynamicDenseData<float>) == sizeof(std::vector<float>));
  STATIC_REQUIRE(sizeof(DynamicDenseData<double>) ==
                 sizeof(std::vector<double>));
}

// ═══════════════════════════════════════════════════════════════════════════
// Layer 2: LinearLayoutExpression / MDArray / MDSpan
//
// LinearLayoutExpression stores m_linear_accessor + m_mapping.
// For fully-static extents, the mapping is empty and
// ZIPPER_NO_UNIQUE_ADDRESS should eliminate it.
// MDArray/MDSpan inherit LinearLayoutExpression with no extra data.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sizeof MDArray (static extents)", "[sizeof][expression][nullary]") {
  using namespace zipper::expression::nullary;
  using zipper::extents;

  // Rank-1: MDArray<T, extents<N>> == N * sizeof(T)
  STATIC_REQUIRE(sizeof(MDArray<float, extents<1>>) == sizeof(float) * 1);
  STATIC_REQUIRE(sizeof(MDArray<float, extents<3>>) == sizeof(float) * 3);
  STATIC_REQUIRE(sizeof(MDArray<float, extents<4>>) == sizeof(float) * 4);
  STATIC_REQUIRE(sizeof(MDArray<double, extents<3>>) == sizeof(double) * 3);

  // Rank-2: MDArray<T, extents<R,C>> == R * C * sizeof(T)
  STATIC_REQUIRE(sizeof(MDArray<float, extents<3, 3>>) == sizeof(float) * 9);
  STATIC_REQUIRE(sizeof(MDArray<double, extents<4, 4>>) ==
                 sizeof(double) * 16);
  STATIC_REQUIRE(sizeof(MDArray<double, extents<2, 3>>) == sizeof(double) * 6);

  // Rank-3
  STATIC_REQUIRE(sizeof(MDArray<float, extents<2, 3, 4>>) ==
                 sizeof(float) * 24);
}

TEST_CASE("sizeof MDSpan (static extents)", "[sizeof][expression][nullary]") {
  using namespace zipper::expression::nullary;
  using zipper::extents;

  // Static-extent MDSpan stores a SpanData (pointer only) + empty mapping.
  STATIC_REQUIRE(sizeof(MDSpan<float, extents<3>>) == sizeof(float *));
  STATIC_REQUIRE(sizeof(MDSpan<double, extents<3>>) == sizeof(double *));
  STATIC_REQUIRE(sizeof(MDSpan<float, extents<3, 3>>) == sizeof(float *));
}

// ═══════════════════════════════════════════════════════════════════════════
// Layer 3: Nullary Expressions
//
// Constant: stores one value_type + static extents (empty base).
// Identity: no data members, static extents via base → 1 byte minimum.
// Unit: stores IndexType (empty for integral_constant, sizeof(index_type)
//       for dynamic index) + static extents via base.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sizeof Constant (static extents)",
          "[sizeof][expression][nullary]") {
  using namespace zipper::expression::nullary;

  // Constant<T, N> stores just m_value; extents are static via base class.
  STATIC_REQUIRE(sizeof(Constant<float, 3>) == sizeof(float));
  STATIC_REQUIRE(sizeof(Constant<double, 3>) == sizeof(double));
  STATIC_REQUIRE(sizeof(Constant<double, 3, 3>) == sizeof(double));
  STATIC_REQUIRE(sizeof(Constant<float>) == sizeof(float));
}

TEST_CASE("sizeof Identity (static extents)",
          "[sizeof][expression][nullary]") {
  using namespace zipper::expression::nullary;

  // Identity has no data members at all — static extents via empty base.
  // Minimum sizeof is 1 for any C++ type.
  STATIC_REQUIRE(sizeof(Identity<float, 3, 3>) == 1);
  STATIC_REQUIRE(sizeof(Identity<double, 3, 3>) == 1);
  STATIC_REQUIRE(sizeof(Identity<float, 4, 4>) == 1);
}

TEST_CASE("sizeof Unit", "[sizeof][expression][nullary]") {
  using namespace zipper::expression::nullary;

  // Static index (integral_constant) + static extent → no data at all.
  using StaticUnit =
      Unit<float, 3, std::integral_constant<zipper::index_type, 1>>;
  STATIC_REQUIRE(sizeof(StaticUnit) == 1);

  // Dynamic index + static extent → stores one index_type.
  using DynamicIndexUnit = Unit<float, 3, zipper::index_type>;
  STATIC_REQUIRE(sizeof(DynamicIndexUnit) == sizeof(zipper::index_type));
}

// ═══════════════════════════════════════════════════════════════════════════
// Layer 4: Top-Level User Types
//
// These wrap ZipperBase<DerivedT, MDArray<...>>.  ZipperBase inherits an
// empty Returnable/NonReturnable mixin (EBO) and stores one Expression
// member.  Total overhead should be zero.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sizeof Vector", "[sizeof][user_types]") {
  // Vector<T, N> should be exactly N * sizeof(T).
  STATIC_REQUIRE(sizeof(zipper::Vector<float, 1>) == sizeof(float) * 1);
  STATIC_REQUIRE(sizeof(zipper::Vector<float, 2>) == sizeof(float) * 2);
  STATIC_REQUIRE(sizeof(zipper::Vector<float, 3>) == sizeof(float) * 3);
  STATIC_REQUIRE(sizeof(zipper::Vector<float, 4>) == sizeof(float) * 4);
  STATIC_REQUIRE(sizeof(zipper::Vector<double, 3>) == sizeof(double) * 3);
  STATIC_REQUIRE(sizeof(zipper::Vector<double, 6>) == sizeof(double) * 6);
}

TEST_CASE("sizeof Matrix", "[sizeof][user_types]") {
  // Matrix<T, R, C> should be exactly R * C * sizeof(T).
  STATIC_REQUIRE(sizeof(zipper::Matrix<float, 2, 2>) == sizeof(float) * 4);
  STATIC_REQUIRE(sizeof(zipper::Matrix<float, 3, 3>) == sizeof(float) * 9);
  STATIC_REQUIRE(sizeof(zipper::Matrix<float, 4, 4>) == sizeof(float) * 16);
  STATIC_REQUIRE(sizeof(zipper::Matrix<double, 3, 3>) == sizeof(double) * 9);
  STATIC_REQUIRE(sizeof(zipper::Matrix<double, 2, 3>) == sizeof(double) * 6);
  STATIC_REQUIRE(sizeof(zipper::Matrix<double, 3, 2>) == sizeof(double) * 6);

  // Column-major should be the same size.
  STATIC_REQUIRE(sizeof(zipper::Matrix<float, 3, 3, false>) ==
                 sizeof(float) * 9);
}

TEST_CASE("sizeof Quaternion", "[sizeof][user_types]") {
  STATIC_REQUIRE(sizeof(zipper::Quaternion<float>) == sizeof(float) * 4);
  STATIC_REQUIRE(sizeof(zipper::Quaternion<double>) == sizeof(double) * 4);
}

TEST_CASE("sizeof Tensor", "[sizeof][user_types]") {
  // Tensor<T, Exts...> should be product(Exts...) * sizeof(T).
  STATIC_REQUIRE(sizeof(zipper::Tensor<float, 3>) == sizeof(float) * 3);
  STATIC_REQUIRE(sizeof(zipper::Tensor<float, 3, 3>) == sizeof(float) * 9);
  STATIC_REQUIRE(sizeof(zipper::Tensor<double, 2, 3, 4>) ==
                 sizeof(double) * 24);
}

TEST_CASE("sizeof Array", "[sizeof][user_types]") {
  STATIC_REQUIRE(sizeof(zipper::Array<float, 3>) == sizeof(float) * 3);
  STATIC_REQUIRE(sizeof(zipper::Array<float, 3, 3>) == sizeof(float) * 9);
  STATIC_REQUIRE(sizeof(zipper::Array<double, 2, 3>) == sizeof(double) * 6);
}

TEST_CASE("sizeof Form", "[sizeof][user_types]") {
  STATIC_REQUIRE(sizeof(zipper::Form<float, 3>) == sizeof(float) * 3);
  STATIC_REQUIRE(sizeof(zipper::Form<double, 3>) == sizeof(double) * 3);
}

TEST_CASE("sizeof DataArray", "[sizeof][user_types]") {
  STATIC_REQUIRE(sizeof(zipper::DataArray<float, 3>) == sizeof(float) * 3);
  STATIC_REQUIRE(sizeof(zipper::DataArray<double, 3>) == sizeof(double) * 3);
}
