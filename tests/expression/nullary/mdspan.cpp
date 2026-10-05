
#include "catch_include.hpp"
#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <mdspan/mdspan.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/MDSpan.hpp>
#include <zipper/expression/nullary/Random.hpp>

namespace {
struct Twice {
    const double *p;
    operator double() const { return 2 * *p; }
};
struct ScalarProxy;
} // namespace

namespace zipper::expression::detail {
template <>
struct ExpressionTraits<ScalarProxy>
    : BasicExpressionTraits<Twice, zipper::extents<>> {
    constexpr static bool is_coefficient_consistent = false;
};
} // namespace zipper::expression::detail

namespace {
struct ScalarProxy : zipper::expression::ExpressionBase<ScalarProxy> {
    explicit ScalarProxy(const double *p) : m_p(p) {}
    auto extents() const -> extents_type { return {}; }
    auto coeff() const -> Twice { return {m_p}; }

    const double *m_p;
};
} // namespace

TEST_CASE("assignment_materializes_scalar_proxy_before_writing",
          "[mdspan][nullary][assignment]") {
    zipper::Matrix<double, 1, 2> a{{7, 7}};
    // Point directly at live storage, not a unary callback's value temporary.
    const ScalarProxy source(&a(0, 0));
    a = source;
    CHECK(a(0, 0) == 14.0);
    CHECK(a(0, 1) == 14.0);
}

TEST_CASE("assignment_snapshots_aliasing_scalar_without_resizing",
          "[mdspan][nullary][assignment]") {
    using namespace zipper;
    auto check = [](auto &destination) {
        destination = expression::nullary::Constant<double>(7.0);
        const auto shape = destination.extents();
        auto tensor = as_tensor(destination);
        auto scalar = tensor.slice(0, 0);
        destination = (scalar * 2.0).expression();
        CHECK(destination.extents() == shape);
        CHECK((destination.as_array() == 14.0).all());
    };
    Matrix<double, 2, 3> fixed;
    MatrixXX<double> dynamic(2, 3);
    check(fixed);
    check(dynamic);
    auto view = dynamic.as_span();
    check(view);
    Tensor<double> scalar;
    scalar() = 7.0;
    scalar = scalar * 2.0;
    CHECK(scalar() == 14.0);
}

TEST_CASE("assignment_rank_zero_generator_remains_per_coefficient",
          "[mdspan][nullary][assignment]") {
    using namespace zipper;
    auto random = expression::nullary::uniform_random<double>(
        extents<>{}, 0.0, 1.0, std::default_random_engine{123});
    auto expected = random;
    MatrixXX<double> destination(2, 3);
    destination = random;
    CHECK(destination.rows() == 2);
    CHECK(destination.cols() == 3);
    // The two sides are separately inlined copies of the same distribution
    // arithmetic; with FMA available (-ffp-contract=fast) the compiler may
    // fuse one and not the other, so they can differ in the last bit. Distinct
    // draws differ by far more, so per-coefficient evaluation is still checked.
    for (index_type i = 0; i < 2; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK_THAT(destination(i, j),
                       Catch::Matchers::WithinULP(expected(), 4));
        }
    }
}

TEST_CASE("test_mdspan_construction", "[mdspan][nullary][dense]") {

  std::array<double, 3> arr = {2, 3, 4};
  std::vector<double> vec = {0, 1, 2};

  std::span<double, 3> span(arr);
  std::span<const double, 3> cspan(arr);
  {
    zipper::expression::nullary::MDSpan<double, zipper::extents<3>> aspan(span);
    zipper::expression::nullary::MDSpan<const double, zipper::extents<3>>
        caspan(cspan);
    STATIC_CHECK(aspan.rank() == 1);
    STATIC_CHECK(aspan.extent(0) == 3);
    REQUIRE(aspan.extents() == zipper::extents<3>{});

    STATIC_CHECK(caspan.rank() == 1);
    STATIC_CHECK(caspan.extent(0) == 3);
    REQUIRE(caspan.extents() == zipper::extents<3>{});

    // check for values first
    for (size_t j = 0; j < 3; ++j) {
      CHECK(aspan(j) == arr[j]);
      CHECK(caspan(j) == arr[j]);
    }

    for (size_t j = 0; j < 3; ++j) {
      CHECK(&aspan.coeff_ref(j) == &arr[j]);
      CHECK(&aspan.const_coeff_ref(j) == &arr[j]);
      CHECK(&caspan.const_coeff_ref(j) == &arr[j]);
      CHECK(&aspan(j) == &arr[j]);

      STATIC_CHECK(decltype(caspan)::traits::is_referrable());
      CHECK(&caspan(j) == &arr[j]);
    }
    for (size_t j = 0; j < 3; ++j) {
      aspan(j) = j + 10;
    }

    for (size_t j = 0; j < 3; ++j) {
      CHECK(aspan(j) == j + 10);
      CHECK(caspan(j) == j + 10);
    }
  }
  {
    std::span<double, std::dynamic_extent> span2(arr);
    std::span<const double, std::dynamic_extent> cspan2(arr);

    zipper::expression::nullary::MDSpan<double,
                                        zipper::extents<std::dynamic_extent>>
        aspan(span2, zipper::extents<3>{});

    zipper::expression::nullary::MDSpan<const double,
                                        zipper::extents<zipper::dynamic_extent>>
        caspan(cspan2, zipper::extents<3>{});
    STATIC_CHECK(aspan.rank() == 1);
    REQUIRE(aspan.extent(0) == 3);
    REQUIRE(aspan.extents() == zipper::extents<std::dynamic_extent>(3));

    STATIC_CHECK(caspan.rank() == 1);
    REQUIRE(caspan.extent(0) == 3);
    REQUIRE(caspan.extents() == zipper::extents<std::dynamic_extent>(3));

    // check for values first
    for (size_t j = 0; j < 3; ++j) {
      CHECK(aspan(j) == arr[j]);
      CHECK(caspan(j) == arr[j]);
    }

    for (size_t j = 0; j < 3; ++j) {
      CHECK(&aspan.coeff_ref(j) == &arr[j]);
      CHECK(&aspan.const_coeff_ref(j) == &arr[j]);
      CHECK(&caspan.const_coeff_ref(j) == &arr[j]);
      CHECK(&aspan(j) == &arr[j]);

      STATIC_CHECK(decltype(caspan)::traits::is_referrable());
      CHECK(&caspan(j) == &arr[j]);
    }
    for (size_t j = 0; j < 3; ++j) {
      aspan(j) = j + 10;
    }

    for (size_t j = 0; j < 3; ++j) {
      CHECK(aspan(j) == j + 10);
      CHECK(caspan(j) == j + 10);
    }

    {
      zipper::expression::nullary::MDSpan<double,
                                          zipper::extents<std::dynamic_extent>>
          aspan(span, zipper::extents<3>{});

      zipper::expression::nullary::MDSpan<
          const double, zipper::extents<zipper::dynamic_extent>>
          caspan(cspan, zipper::extents<3>{});
      STATIC_CHECK(aspan.rank() == 1);
      REQUIRE(aspan.extent(0) == 3);
      REQUIRE(aspan.extents() == zipper::extents<std::dynamic_extent>(3));

      STATIC_CHECK(caspan.rank() == 1);
      REQUIRE(caspan.extent(0) == 3);
      REQUIRE(caspan.extents() == zipper::extents<std::dynamic_extent>(3));
    }
  }
}

TEST_CASE("mdspan_static_extents_from_dynamic_span",
          "[mdspan][nullary][dense]") {
  using namespace zipper;
  using Vec3Span = expression::nullary::MDSpan<int, extents<3>>;
  using ScalarSpan = expression::nullary::MDSpan<int, extents<>>;

  // Only explicit construction is allowed from a dynamic-extent span.
  STATIC_CHECK(std::is_constructible_v<Vec3Span, std::span<int>>);
  STATIC_CHECK_FALSE(std::is_convertible_v<std::span<int>, Vec3Span>);
  STATIC_CHECK(std::is_constructible_v<ScalarSpan, std::span<int>>);
  STATIC_CHECK_FALSE(std::is_convertible_v<std::span<int>, ScalarSpan>);

  std::array<int, 3> data{1, 2, 3};
  const Vec3Span v(std::span<int>(data.data(), data.size()));
  CHECK(v(0) == 1);
  CHECK(v(2) == 3);

  const ScalarSpan s(std::span<int>(data.data() + 1, 1));
  CHECK(s() == 2);
}
