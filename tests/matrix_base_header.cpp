#include <zipper/MatrixBase.hpp>

#include <catch2/catch_test_macros.hpp>

#include <vector>

TEST_CASE("matrix_base_header_exposes_mdspan_ctad", "[matrix][mdspan][ctad]") {
  std::vector<double> data(6);
  zipper::mdspan<double, zipper::dextents<2>> span(data.data(), 2, 3);
  zipper::MatrixBase matrix(span);

  matrix(1, 2) = 7;
  CHECK(data[5] == 7);
}
