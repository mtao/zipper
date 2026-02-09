
#include "catch_include.hpp"
#include <zipper/Container.hpp>
#include <zipper/Vector.hpp>
#include <zipper/types.hpp>

using namespace zipper;

// ===== Basic construction and element access =====

TEST_CASE("container_basic_construction", "[container]") {
  Container<double, 3> c;
  c(0) = 1.0;
  c(1) = 2.0;
  c(2) = 3.0;

  CHECK(c(0) == 1.0);
  CHECK(c(1) == 2.0);
  CHECK(c(2) == 3.0);
  REQUIRE(c.extent(0) == 3);
}

TEST_CASE("container_2d_construction", "[container]") {
  Container<int, 2, 3> c;
  for (index_type i = 0; i < 2; ++i) {
    for (index_type j = 0; j < 3; ++j) {
      c(i, j) = static_cast<int>(i * 3 + j);
    }
  }

  CHECK(c(0, 0) == 0);
  CHECK(c(0, 1) == 1);
  CHECK(c(0, 2) == 2);
  CHECK(c(1, 0) == 3);
  CHECK(c(1, 1) == 4);
  CHECK(c(1, 2) == 5);
}

// ===== Owning Container: data(), begin()/end(), as_span() =====

TEST_CASE("container_data_access", "[container]") {
  Container<double, 4> c;
  c(0) = 10.0;
  c(1) = 20.0;
  c(2) = 30.0;
  c(3) = 40.0;

  auto *ptr = c.data();
  REQUIRE(ptr != nullptr);
  // Data should reflect the values set
  // (layout-dependent, but for rank-1 left-major it's sequential)
  CHECK(ptr[0] == 10.0);
  CHECK(ptr[1] == 20.0);
  CHECK(ptr[2] == 30.0);
  CHECK(ptr[3] == 40.0);
}

TEST_CASE("container_begin_end", "[container]") {
  Container<int, 3> c;
  c(0) = 5;
  c(1) = 10;
  c(2) = 15;

  std::vector<int> vals(c.begin(), c.end());
  REQUIRE(vals.size() == 3);
  CHECK(vals[0] == 5);
  CHECK(vals[1] == 10);
  CHECK(vals[2] == 15);
}

TEST_CASE("container_as_span", "[container]") {
  Container<double, 3> c;
  c(0) = 1.0;
  c(1) = 2.0;
  c(2) = 3.0;

  auto sp = c.as_span();
  REQUIRE(sp.extent(0) == 3);
  CHECK(sp(0) == 1.0);
  CHECK(sp(1) == 2.0);
  CHECK(sp(2) == 3.0);

  // Non-owning span should see mutations
  c(1) = 99.0;
  CHECK(sp(1) == 99.0);
}

// ===== operator<=> and operator== =====

TEST_CASE("container_comparison_equal", "[container][comparison]") {
  Container<int, 3> a;
  a(0) = 1;
  a(1) = 2;
  a(2) = 3;

  Container<int, 3> b;
  b(0) = 1;
  b(1) = 2;
  b(2) = 3;

  CHECK(a == b);
  CHECK((a <=> b) == std::weak_ordering::equivalent);
}

TEST_CASE("container_comparison_less", "[container][comparison]") {
  Container<int, 3> a;
  a(0) = 1;
  a(1) = 2;
  a(2) = 3;

  Container<int, 3> b;
  b(0) = 1;
  b(1) = 2;
  b(2) = 4;

  CHECK((a <=> b) == std::weak_ordering::less);
  CHECK((b <=> a) == std::weak_ordering::greater);
}

TEST_CASE("container_comparison_different_extents", "[container][comparison]") {
  // Dynamic extents: different sizes
  detail::Container_<int, dextents<1>> a(3);
  a(0) = 1;
  a(1) = 2;
  a(2) = 3;

  detail::Container_<int, dextents<1>> b(4);
  b(0) = 1;
  b(1) = 2;
  b(2) = 3;
  b(3) = 4;

  // a has fewer elements with same prefix — a's extent(0)=3 < b's extent(0)=4
  CHECK((a <=> b) == std::weak_ordering::less);
  CHECK((b <=> a) == std::weak_ordering::greater);
}

TEST_CASE("container_comparison_2d", "[container][comparison]") {
  Container<int, 2, 2> a;
  a(0, 0) = 1;
  a(0, 1) = 2;
  a(1, 0) = 3;
  a(1, 1) = 4;

  Container<int, 2, 2> b;
  b(0, 0) = 1;
  b(0, 1) = 2;
  b(1, 0) = 3;
  b(1, 1) = 4;

  CHECK(a == b);

  b(1, 1) = 5;
  CHECK((a <=> b) == std::weak_ordering::less);
}

// ===== Slicing =====

TEST_CASE("container_slice_full_extent", "[container][slice]") {
  Container<double, 4> c;
  c(0) = 10.0;
  c(1) = 20.0;
  c(2) = 30.0;
  c(3) = 40.0;

  auto s = c.slice<full_extent_t>();
  REQUIRE(s.extent(0) == 4);
  CHECK(s(0) == 10.0);
  CHECK(s(1) == 20.0);
  CHECK(s(2) == 30.0);
  CHECK(s(3) == 40.0);
}

TEST_CASE("container_slice_subrange", "[container][slice]") {
  Container<double, 5> c;
  for (index_type i = 0; i < 5; ++i) {
    c(i) = static_cast<double>(i * 10);
  }

  // Slice elements 1..3 (offset=1, extent=3)
  auto s = c.slice(zipper::slice(1, 3));
  REQUIRE(s.extent(0) == 3);
  CHECK(s(0) == 10.0);
  CHECK(s(1) == 20.0);
  CHECK(s(2) == 30.0);
}

TEST_CASE("container_slice_static", "[container][slice]") {
  Container<double, 5> c;
  for (index_type i = 0; i < 5; ++i) {
    c(i) = static_cast<double>(i);
  }

  auto s = c.slice<static_slice_t<1, 3>>();
  using slice_extents = typename std::decay_t<decltype(s)>::extents_type;
  static_assert(slice_extents::static_extent(0) == 3);
  REQUIRE(s.extent(0) == 3);
  CHECK(s(0) == 1.0);
  CHECK(s(1) == 2.0);
  CHECK(s(2) == 3.0);
}

TEST_CASE("container_slice_2d", "[container][slice]") {
  Container<int, 3, 4> c;
  for (index_type i = 0; i < 3; ++i) {
    for (index_type j = 0; j < 4; ++j) {
      c(i, j) = static_cast<int>(i * 4 + j);
    }
  }

  // Take the full first dimension, columns 1..2
  auto s = c.slice(full_extent_t{}, zipper::slice(1, 2));
  REQUIRE(s.extent(0) == 3);
  REQUIRE(s.extent(1) == 2);
  CHECK(s(0, 0) == 1);
  CHECK(s(0, 1) == 2);
  CHECK(s(1, 0) == 5);
  CHECK(s(1, 1) == 6);
  CHECK(s(2, 0) == 9);
  CHECK(s(2, 1) == 10);
}

TEST_CASE("container_slice_mutable", "[container][slice]") {
  Container<double, 4> c;
  for (index_type i = 0; i < 4; ++i) {
    c(i) = static_cast<double>(i);
  }

  // Mutable slice — writing through it should modify the original
  auto s = c.slice(zipper::slice(1, 2));
  s(0) = 99.0;
  s(1) = 88.0;

  CHECK(c(0) == 0.0);
  CHECK(c(1) == 99.0);
  CHECK(c(2) == 88.0);
  CHECK(c(3) == 3.0);
}

// ===== Swizzle =====

TEST_CASE("container_swizzle_transpose_2d", "[container][swizzle]") {
  Container<int, 2, 3> c;
  for (index_type i = 0; i < 2; ++i) {
    for (index_type j = 0; j < 3; ++j) {
      c(i, j) = static_cast<int>(i * 3 + j);
    }
  }

  // Swizzle<1,0> is a transpose
  auto t = c.swizzle<ContainerBase, 1, 0>();
  REQUIRE(t.extent(0) == 3);
  REQUIRE(t.extent(1) == 2);
  for (index_type i = 0; i < 2; ++i) {
    for (index_type j = 0; j < 3; ++j) {
      CHECK(t(j, i) == c(i, j));
    }
  }
}

// ===== Cast =====

TEST_CASE("container_cast", "[container][cast]") {
  Container<int, 3> c;
  c(0) = 1;
  c(1) = 2;
  c(2) = 3;

  auto d = c.cast<double>();
  static_assert(
      std::is_same_v<typename std::decay_t<decltype(d)>::value_type, double>);
  CHECK(d(0) == 1.0);
  CHECK(d(1) == 2.0);
  CHECK(d(2) == 3.0);
}

// ===== eval() =====

TEST_CASE("container_eval", "[container]") {
  Container<double, 3> c;
  c(0) = 1.0;
  c(1) = 2.0;
  c(2) = 3.0;

  auto e = c.eval();
  // eval() should produce an owning copy
  CHECK(e(0) == 1.0);
  CHECK(e(1) == 2.0);
  CHECK(e(2) == 3.0);

  // Modifying original should not affect eval'd copy
  c(0) = 999.0;
  CHECK(e(0) == 1.0);
}

// ===== as_array() bridge =====

TEST_CASE("container_as_array", "[container]") {
  Container<double, 3> c;
  c(0) = 1.0;
  c(1) = 2.0;
  c(2) = 3.0;

  auto a = c.as_array();
  CHECK(a(0) == 1.0);
  CHECK(a(1) == 2.0);
  CHECK(a(2) == 3.0);
}

// ===== ContainerBase wrapping MDSpan (non-owning) =====

TEST_CASE("containerbase_from_span", "[container][mdspan]") {
  std::array<double, 4> data = {10.0, 20.0, 30.0, 40.0};
  std::span<double, 4> sp(data);
  ContainerBase cb(sp);

  REQUIRE(cb.extent(0) == 4);
  CHECK(cb(0) == 10.0);
  CHECK(cb(1) == 20.0);
  CHECK(cb(2) == 30.0);
  CHECK(cb(3) == 40.0);

  // Modifying underlying data is visible through the wrapper
  data[2] = 999.0;
  CHECK(cb(2) == 999.0);
}

TEST_CASE("containerbase_from_vector", "[container][mdspan]") {
  std::vector<int> data = {5, 10, 15};
  ContainerBase cb(data);

  REQUIRE(cb.extent(0) == 3);
  CHECK(cb(0) == 5);
  CHECK(cb(1) == 10);
  CHECK(cb(2) == 15);
}

TEST_CASE("containerbase_from_array", "[container][mdspan]") {
  std::array<int, 3> data = {1, 2, 3};
  ContainerBase cb(data);

  REQUIRE(cb.extent(0) == 3);
  CHECK(cb(0) == 1);
  CHECK(cb(1) == 2);
  CHECK(cb(2) == 3);
}

// ===== ContainerBase slice on non-owning view =====

TEST_CASE("containerbase_slice_nonowning", "[container][slice][mdspan]") {
  std::array<double, 5> data = {0.0, 1.0, 2.0, 3.0, 4.0};
  std::span<double, 5> sp(data);
  ContainerBase cb(sp);

  auto s = cb.slice(zipper::slice(1, 3));
  REQUIRE(s.extent(0) == 3);
  CHECK(s(0) == 1.0);
  CHECK(s(1) == 2.0);
  CHECK(s(2) == 3.0);
}

// ===== Dynamic extent Container =====

TEST_CASE("container_dynamic", "[container]") {
  detail::Container_<double, dextents<1>> c(5);
  for (index_type i = 0; i < 5; ++i) {
    c(i) = static_cast<double>(i * 2);
  }

  CHECK(c(0) == 0.0);
  CHECK(c(1) == 2.0);
  CHECK(c(2) == 4.0);
  CHECK(c(3) == 6.0);
  CHECK(c(4) == 8.0);

  auto s = c.slice(zipper::slice(2, 2));
  REQUIRE(s.extent(0) == 2);
  CHECK(s(0) == 4.0);
  CHECK(s(1) == 6.0);
}
