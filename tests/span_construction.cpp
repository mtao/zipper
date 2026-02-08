#include <numeric>
#include <zipper/Array.hpp>
#include <zipper/ArrayBase.hxx>
#include <zipper/Form.hpp>
#include <zipper/FormBase.hxx>
#include <zipper/Matrix.hpp>
#include <zipper/MatrixBase.hxx>
#include <zipper/Tensor.hpp>
#include <zipper/TensorBase.hxx>
#include <zipper/Vector.hpp>
#include <zipper/VectorBase.hxx>

#include "catch_include.hpp"

TEST_CASE("test_span_construction", "[vector][storage][dense]") {
  // Static vector span from std::array
  {
    std::array<double, 3> data;
    data[0] = 10;
    data[1] = 20;
    data[2] = 30;
    auto sp = std::span<double, 3>(data);
    zipper::Vector<double, 3>::span_type v{sp};

    CHECK(&data[0] == &v(0));
    CHECK(&data[1] == &v(1));
    CHECK(&data[2] == &v(2));
    CHECK(v(0) == 10);
    CHECK(v(1) == 20);
    CHECK(v(2) == 30);
  }

  // Multiple rank-1 span types wrapping same std::array
  {
    std::array<double, 3> data;
    data[0] = 1;
    data[1] = 2;
    data[2] = 3;
    auto sp = std::span<double, 3>(data);
    zipper::Vector<double, 3>::span_type v{sp};
    zipper::Tensor<double, 3>::span_type T{sp};
    zipper::Form<double, 3>::span_type F{sp};
    zipper::Array<double, 3>::span_type A{sp};

    REQUIRE(v.extent(0) == data.size());
    static_assert(std::decay_t<decltype(v.extents())>::static_extent(0) ==
                  data.size());
    CHECK(&data[0] == &v(0));
    CHECK(&data[1] == &v(1));
    CHECK(&data[2] == &v(2));

    CHECK(&data[0] == &F(0));
    CHECK(&data[1] == &F(1));
    CHECK(&data[2] == &F(2));

    CHECK(&data[0] == &T(0));
    CHECK(&data[1] == &T(1));
    CHECK(&data[2] == &T(2));

    CHECK(&data[0] == &A(0));
    CHECK(&data[1] == &A(1));
    CHECK(&data[2] == &A(2));
  }

  // Static 2D span types (4x5 row-major) from std::array
  {
    std::array<double, 20> data;
    std::iota(data.begin(), data.end(), 0);

    auto sp = std::span<double, 20>(data);
    zipper::Matrix<double, 4, 5>::span_type M{sp};
    zipper::Tensor<double, 4, 5>::span_type T{sp};
    zipper::Form<double, 4, 5>::span_type F{sp};
    zipper::Array<double, 4, 5>::span_type A{sp};
    for (size_t j = 0; j < 4; ++j) {
      for (size_t k = 0; k < 5; ++k) {
        CHECK(&data[j * 5 + k] == &M(j, k));
        CHECK(&data[j * 5 + k] == &T(j, k));
        CHECK(&data[j * 5 + k] == &F(j, k));
        CHECK(&data[j * 5 + k] == &A(j, k));
      }
    }
  }

  // Column-major 2D span types (4x5, LeftMajor=false for Matrix)
  {
    std::vector<double> data(20);
    std::iota(data.begin(), data.end(), 0);

    auto sp = std::span<double, 20>(data.data(), 20);
    zipper::Matrix<double, 4, 5, false>::span_type M{sp};
    zipper::Tensor_<double, zipper::extents<4, 5>, false>::span_type T{sp};
    for (size_t j = 0; j < 4; ++j) {
      for (size_t k = 0; k < 5; ++k) {
        CHECK(&data[j + k * 4] == &M(j, k));
        CHECK(&data[j + k * 4] == &T(j, k));
      }
    }
  }

  // Row-major 2D span types - verify pointer values
  {
    std::vector<double> data(20);
    std::iota(data.begin(), data.end(), 0);

    auto sp = std::span<double, 20>(data.data(), 20);
    zipper::Matrix<double, 4, 5, true>::span_type M{sp};
    zipper::Tensor_<double, zipper::extents<4, 5>, true>::span_type T{sp};

    for (size_t j = 0; j < 4; ++j) {
      for (size_t k = 0; k < 5; ++k) {
        CHECK(&data[j * 5 + k] == &M(j, k));
        CHECK(&data[j * 5 + k] == &T(j, k));
      }
    }
  }

  // Span writability: writing through span modifies original data
  {
    std::array<double, 3> data = {0, 0, 0};
    auto sp = std::span<double, 3>(data);
    zipper::Vector<double, 3>::span_type v{sp};

    v(0) = 10;
    v(1) = 20;
    v(2) = 30;

    CHECK(data[0] == 10);
    CHECK(data[1] == 20);
    CHECK(data[2] == 30);
  }

  // as_span() from owning types
  {
    zipper::Vector<double, 3> v = {1.0, 2.0, 3.0};
    auto span = v.as_span();

    CHECK(&v(0) == &span(0));
    CHECK(&v(1) == &span(1));
    CHECK(&v(2) == &span(2));

    span(0) = 10;
    CHECK(v(0) == 10);
  }
  {
    zipper::Matrix<double, 2, 3> M;
    M.row(0) = {1.0, 2.0, 3.0};
    M.row(1) = {4.0, 5.0, 6.0};
    auto span = M.as_span();

    CHECK(&M(0, 0) == &span(0, 0));
    CHECK(&M(1, 2) == &span(1, 2));

    span(0, 0) = 99;
    CHECK(M(0, 0) == 99);
  }
}

TEST_CASE("test_mdspan_deduction_guides", "[mdspan][ctad]") {
  // Rank-1 mdspan → VectorBase, TensorBase, FormBase, ArrayBase via CTAD
  {
    std::array<double, 3> data = {10, 20, 30};
    zipper::mdspan<double, zipper::extents<3>> ms(data.data());

    zipper::VectorBase v(ms);
    zipper::TensorBase t(ms);
    zipper::FormBase f(ms);
    zipper::ArrayBase a(ms);

    // Verify these are views (pointer equality with original data)
    CHECK(&data[0] == &v(0));
    CHECK(&data[1] == &v(1));
    CHECK(&data[2] == &v(2));

    CHECK(&data[0] == &t(0));
    CHECK(&data[1] == &t(1));
    CHECK(&data[2] == &t(2));

    CHECK(&data[0] == &f(0));
    CHECK(&data[1] == &f(1));
    CHECK(&data[2] == &f(2));

    CHECK(&data[0] == &a(0));
    CHECK(&data[1] == &a(1));
    CHECK(&data[2] == &a(2));

    // Verify values
    CHECK(v(0) == 10);
    CHECK(v(1) == 20);
    CHECK(v(2) == 30);

    // Verify writability through the view
    v(0) = 99;
    CHECK(data[0] == 99);
    CHECK(t(0) == 99);
  }

  // Rank-2 mdspan → MatrixBase, TensorBase, FormBase, ArrayBase via CTAD
  {
    std::array<double, 6> data;
    std::iota(data.begin(), data.end(), 1);  // {1,2,3,4,5,6}
    zipper::mdspan<double, zipper::extents<2, 3>> ms(data.data());

    zipper::MatrixBase m(ms);
    zipper::TensorBase t(ms);
    zipper::FormBase f(ms);
    zipper::ArrayBase a(ms);

    // Verify pointer equality (views, not copies)
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        CHECK(&data[i * 3 + j] == &m(i, j));
        CHECK(&data[i * 3 + j] == &t(i, j));
        CHECK(&data[i * 3 + j] == &f(i, j));
        CHECK(&data[i * 3 + j] == &a(i, j));
      }
    }

    // Verify values
    CHECK(m(0, 0) == 1);
    CHECK(m(0, 2) == 3);
    CHECK(m(1, 0) == 4);
    CHECK(m(1, 2) == 6);

    // Verify writability
    m(1, 1) = 99;
    CHECK(data[4] == 99);
    CHECK(t(1, 1) == 99);
  }

  // Const mdspan → read-only views via CTAD
  {
    const std::array<double, 3> data = {5, 10, 15};
    zipper::mdspan<const double, zipper::extents<3>> ms(data.data());

    zipper::VectorBase v(ms);
    zipper::TensorBase t(ms);
    zipper::FormBase f(ms);
    zipper::ArrayBase a(ms);

    CHECK(v(0) == 5);
    CHECK(v(1) == 10);
    CHECK(v(2) == 15);

    CHECK(t(0) == 5);
    CHECK(f(0) == 5);
    CHECK(a(0) == 5);
  }
}
