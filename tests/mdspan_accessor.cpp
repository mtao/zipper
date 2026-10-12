#include <zipper/Matrix.hpp>
#include <zipper/MatrixBase.hxx>
#include <zipper/Vector.hpp>
#include <zipper/VectorBase.hxx>

#include "catch_include.hpp"

namespace {

// Read-only accessor returning a computed value (like
// std::linalg::scaled_accessor).
template <typename T>
struct scaled_accessor {
  using element_type = T;
  using reference = std::remove_const_t<T>;
  using data_handle_type = T *;
  using offset_policy = scaled_accessor;

  scaled_accessor() = default;
  explicit scaled_accessor(std::remove_const_t<T> s) : scale(s) {}

  auto access(data_handle_type p, std::size_t i) const -> reference {
    return scale * p[i];
  }
  auto offset(data_handle_type p, std::size_t i) const -> data_handle_type {
    return p + i;
  }

  std::remove_const_t<T> scale = 1;
};

// Reference-returning accessor that is not default_accessor: element i
// lives at p[i + shift].
template <typename T>
struct shifted_accessor {
  using element_type = T;
  using reference = T &;
  using data_handle_type = T *;
  using offset_policy = shifted_accessor;

  auto access(data_handle_type p, std::size_t i) const -> reference {
    return p[i + shift];
  }
  auto offset(data_handle_type p, std::size_t i) const -> data_handle_type {
    return p + i;
  }

  std::size_t shift = 0;
};

using E23 = zipper::extents<2, 3>;
using Layout = zipper::storage::row_major;

} // namespace

TEST_CASE("mdspan_proxy_accessor_is_honored", "[mdspan][accessor]") {
  const std::array<double, 6> data{1, 2, 3, 4, 5, 6};
  zipper::mdspan<const double, E23, Layout, scaled_accessor<const double>> m(
      data.data(), Layout::mapping<E23>{}, scaled_accessor<const double>(2.0));

  zipper::MatrixBase M(m);
  CHECK(M(0, 0) == 2.0);
  CHECK(M(1, 2) == 12.0);

  // Composes with other expressions.
  zipper::Vector<double, 3> x{1.0, 0.0, 1.0};
  zipper::Vector<double, 2> y = M * x;
  CHECK(y(0) == 8.0);
  CHECK(y(1) == 20.0);

  // A proxy reference is neither writable nor a raw linear array.
  using traits = std::decay_t<decltype(M.expression())>::traits;
  STATIC_CHECK_FALSE(traits::is_assignable());
  STATIC_CHECK_FALSE(traits::is_linear_array);
}

TEST_CASE("mdspan_reference_accessor_is_writable", "[mdspan][accessor]") {
  std::array<double, 7> data{};
  zipper::mdspan<double, E23, Layout, shifted_accessor<double>> m(
      data.data(), Layout::mapping<E23>{}, shifted_accessor<double>{1});

  zipper::MatrixBase M(m);
  M = zipper::Matrix<double, 2, 3>{{1, 2, 3}, {4, 5, 6}};
  CHECK(data[0] == 0.0);
  CHECK(data[1] == 1.0);
  CHECK(data[6] == 6.0);
  CHECK(M(1, 0) == 4.0);
}

TEST_CASE("as_mdspan_accessor_choice", "[mdspan][accessor]") {
  zipper::Matrix<double, 2, 3> A{{1, 2, 3}, {4, 5, 6}};

  // Default accessor, mutable and const.
  auto m = A.expression().as_mdspan();
  STATIC_CHECK(std::is_same_v<decltype(m)::accessor_type,
                              zipper::default_accessor_policy<double>>);
  m[0, 1] = 7.0;
  CHECK(A(0, 1) == 7.0);

  const auto &C = A;
  auto c = C.expression().as_mdspan();
  STATIC_CHECK(std::is_same_v<decltype(c)::accessor_type,
                              zipper::default_accessor_policy<const double>>);
  CHECK(&c[1, 2] == &A(1, 2));

  // Caller-chosen accessor; round-trips back through zipper.
  auto s = C.expression().as_mdspan(scaled_accessor<const double>(10.0));
  CHECK(s[1, 0] == 40.0);
  zipper::MatrixBase S(s);
  CHECK(S(0, 1) == 70.0);
}
