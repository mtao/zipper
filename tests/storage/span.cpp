
#include <iostream>
#include <span>
#include <zipper/storage/SpanData.hpp>

#include "../catch_include.hpp"
using namespace zipper;
TEST_CASE("test_span_1d", "[storage][dense][span]") {
  std::vector<int> vec = {2, 3};
  std::span<int, std::dynamic_extent> svec = vec;
  storage::SpanData<int, zipper::dynamic_extent> v(svec);
  storage::SpanData<const int, zipper::dynamic_extent> v_const(svec);

  storage::SpanData<int, 2> v2(svec);
  storage::SpanData<const int, 2> v2_const(svec);

  CHECK(v.coeff(0) == 2);
  CHECK(v.coeff(1) == 3);

  std::array<int, 2> y;
  storage::SpanData<int, 2> z(y);
  storage::SpanData<const int, 2> z_const(y);
  CHECK(y[0] == 2);
  CHECK(y[1] == 3);

  z.coeff_ref(0) = 3;
  z.coeff_ref(1) = 4;

  CHECK(y[0] == 3);
  CHECK(y[1] == 4);

  z.coeff_ref(0) = 3;
  z.coeff_ref(1) = 4;

  CHECK(y[0] == 3);
  CHECK(y[1] == 4);
  CHECK(z_const.coeff(0) == 3);
  CHECK(z_const.coeff(1) == 4);
  CHECK(z_const.const_coeff_ref(0) == 3);
  CHECK(z_const.const_coeff_ref(1) == 4);
}
