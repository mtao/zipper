
#include "catch_include.hpp"
#include "fmt_include.hpp"
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/MDSpan.hpp>
#include <zipper/expression/nullary/Random.hpp>
TEST_CASE("test_mdspan_construction", "[mdspan][nullary][dense]") {

  std::array<double, 3> arr = {2, 3, 4};
  std::vector<double> vec = {0, 1, 2};

  std::span<double, 3> span(arr);
  std::span<const double, 3> cspan(arr);
  {
    zipper::expression::nullary::MDSpan<double, zipper::extents<3>> aspan(span);
    zipper::expression::nullary::MDSpan<const double, zipper::extents<3>>
        caspan(cspan);
    static_assert(aspan.rank() == 1);
    static_assert(aspan.extent(0) == 3);
    static_assert(aspan.extents() == zipper::extents<3>{});

    static_assert(caspan.rank() == 1);
    static_assert(caspan.extent(0) == 3);
    static_assert(caspan.extents() == zipper::extents<3>{});

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

      static_assert(decltype(caspan)::traits::is_referrable());
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
}
