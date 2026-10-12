#include <zipper/Matrix.hpp>
#include <zipper/MatrixBase.hxx>
#include <zipper/Vector.hpp>
#include <zipper/VectorBase.hxx>

#include <numeric>

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

  // Static size: only the handle and the accessor are stored.
  using storage_t = std::decay_t<decltype(M.expression().linear_accessor())>;
  STATIC_CHECK(sizeof(storage_t) == sizeof(const double *) + sizeof(double));

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

namespace {
using Stride = zipper::storage::layout_stride;
// A 2x3 view of every other element of a 12-element buffer, column-major:
// (i, j) -> 2*i + 4*j, so it reaches index 10 (past 2*3 = 6 elements).
auto strided_mapping() -> Stride::mapping<E23> {
  return {E23{}, std::array<std::size_t, 2>{2, 4}};
}
} // namespace

TEST_CASE("mdspan_layout_stride_read_write", "[mdspan][layout_stride]") {
  std::array<double, 12> data{};
  std::iota(data.begin(), data.end(), 0.0);
  zipper::mdspan<double, E23, Stride> m(data.data(), strided_mapping());

  zipper::MatrixBase M(m);
  CHECK(M(0, 0) == 0.0);
  CHECK(M(1, 0) == 2.0);
  CHECK(M(0, 1) == 4.0);
  CHECK(M(1, 2) == 10.0);

  // Strides survive through the expression and back out.
  auto back = M.expression().as_mdspan();
  STATIC_CHECK(std::is_same_v<decltype(back)::layout_type, Stride>);
  CHECK(back.stride(0) == 2);
  CHECK(back.stride(1) == 4);
  CHECK(&back[1, 2] == &data[10]);

  // Kernels see the right values.
  zipper::Vector<double, 3> x{1.0, 1.0, 1.0};
  zipper::Vector<double, 2> y = M * x;
  CHECK(y(0) == 0.0 + 4.0 + 8.0);
  CHECK(y(1) == 2.0 + 6.0 + 10.0);
  zipper::Matrix<double, 3, 2> Mt = M.transpose();
  CHECK(Mt(2, 1) == 10.0);

  // Writes land on the strided elements only.
  M = zipper::Matrix<double, 2, 3>{{-1, -2, -3}, {-4, -5, -6}};
  CHECK(data[0] == -1.0);
  CHECK(data[2] == -4.0);
  CHECK(data[10] == -6.0);
  CHECK(data[1] == 1.0);
  CHECK(data[11] == 11.0);
}

TEST_CASE("mdspan_layout_stride_eval", "[mdspan][layout_stride]") {
  std::array<double, 12> data{};
  std::iota(data.begin(), data.end(), 0.0);
  zipper::mdspan<const double, E23, Stride> m(data.data(), strided_mapping());
  zipper::MatrixBase M(m);

  // Owned copies use a dense layout.
  auto E = M.eval();
  CHECK(E(1, 2) == 10.0);
  auto O = M.expression().make_owned();
  STATIC_CHECK_FALSE(
      std::is_same_v<std::decay_t<decltype(O)>::layout_policy, Stride>);
  CHECK(O(0, 1) == 4.0);

  // Dynamic extents with a strided mapping, viewed via the (ptr, mapping)
  // constructor.
  using DE = zipper::dextents<2>;
  zipper::expression::nullary::MDSpan<const double, DE, Stride> D(
      data.data(), Stride::mapping<DE>{DE{3, 2}, std::array<std::size_t, 2>{1, 6}});
  CHECK(zipper::MatrixBase(D)(2, 1) == 8.0);
}
