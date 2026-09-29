#include "catch_include.hpp"

#include <zipper/Array.hpp>
#include <zipper/DataArray.hpp>
#include <zipper/Form.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Quaternion.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/storage/DenseData.hpp>

#include <algorithm>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>

using namespace zipper;

namespace {
struct NonTrivial {
  int value = 42;
};

// True iff `{}` can copy-list-initialize a T (i.e. `T t = {};` compiles).
template <typename T>
concept CopyListInitFromEmpty = requires { [](T) {}({}); };

template <typename T> auto all_zero(const T &range) -> bool {
  return std::ranges::all_of(range, [](const auto &x) { return x == 0; });
}
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Tag contract
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("uninitialized_tag_contract", "[uninitialized]") {
  STATIC_CHECK(std::is_empty_v<uninitialized_t>);
  STATIC_CHECK(std::is_same_v<std::remove_cv_t<decltype(uninitialized)>,
                              uninitialized_t>);
  // `{}` must not silently select the tag overload.
  STATIC_CHECK_FALSE(CopyListInitFromEmpty<uninitialized_t>);
  STATIC_CHECK(CopyListInitFromEmpty<index_type>); // sanity-check the probe
  // The tag never converts into an owning type.
  STATIC_CHECK_FALSE(std::is_convertible_v<uninitialized_t, Vector<double, 3>>);
  STATIC_CHECK_FALSE(
      std::is_convertible_v<uninitialized_t, Matrix<double, 3, 3>>);
}

TEST_CASE("uninitialized_non_trivial_types_are_default_constructed",
          "[uninitialized]") {
  // The tag means default-initialization: accepted for any
  // default-constructible type; non-trivial types are default-constructed.
  STATIC_CHECK(std::is_constructible_v<storage::DenseData<std::string, 3>,
                                       uninitialized_t>);
  STATIC_CHECK(std::is_constructible_v<
               storage::DenseData<std::string, std::dynamic_extent>,
               uninitialized_t, index_type>);
  STATIC_CHECK(std::is_constructible_v<VectorX<std::string>, uninitialized_t,
                                       index_type>);
  STATIC_CHECK(
      std::is_constructible_v<Vector<std::string, 3>, uninitialized_t>);
  STATIC_CHECK(std::is_constructible_v<MatrixXX<std::string>, uninitialized_t,
                                       index_type, index_type>);
  STATIC_CHECK(
      std::is_constructible_v<Array<NonTrivial, 3>, uninitialized_t>);

  VectorX<std::string> s(uninitialized, 3);
  CHECK(s(0).empty());
  CHECK(s(2).empty());
  Vector<std::string, 2> ss(uninitialized);
  CHECK(ss(1).empty());
  storage::DenseData<NonTrivial, std::dynamic_extent> d(uninitialized, 4);
  CHECK(d.coeff(3).value == 42);
  storage::DenseData<NonTrivial, 3> ds(uninitialized);
  CHECK(ds.coeff(0).value == 42);

  VectorX<std::string> grow(uninitialized, 1);
  grow(0) = "keep";
  grow.resize(uninitialized, 5);
  CHECK(grow(0) == "keep");
  CHECK(grow(4).empty());
}

TEST_CASE("uninitialized_static_types_stay_trivially_copyable",
          "[uninitialized][trivially_copyable]") {
  STATIC_CHECK(std::is_trivially_copyable_v<storage::DenseData<double, 3>>);
  STATIC_CHECK(std::is_trivially_copyable_v<Vector<double, 3>>);
  STATIC_CHECK(std::is_trivially_copyable_v<Matrix<double, 3, 3>>);
  STATIC_CHECK(std::is_trivially_copyable_v<Quaternion<double>>);
}

// ─────────────────────────────────────────────────────────────────────────────
// Default (plain) construction still zero-fills
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("plain_construction_zero_fills", "[uninitialized][regression]") {
  CHECK(all_zero(Vector<double, 3>{}));
  CHECK(all_zero(Vector<double, 3>()));
  CHECK(all_zero(VectorX<double>(257)));

  MatrixXX<double> m(17, 19);
  CHECK(all_zero(m.expression().linear_accessor()));
  Matrix<double, 4, 4> ms;
  CHECK(all_zero(ms.expression().linear_accessor()));

  storage::DenseData<double, std::dynamic_extent> d(64);
  CHECK(all_zero(d));
  storage::DenseData<double, 5> ds{};
  CHECK(all_zero(ds));
}

TEST_CASE("plain_resize_zero_fills_new_elements",
          "[uninitialized][regression]") {
  VectorX<double> v(4);
  std::ranges::fill(v, 7.0);
  v.resize(10);
  CHECK(v.size() == 10);
  for (index_type i = 0; i < 4; ++i) {
    CHECK(v(i) == 7.0);
  }
  for (index_type i = 4; i < 10; ++i) {
    CHECK(v(i) == 0.0);
  }

  // Shrink then grow within capacity: the regrown tail must be re-zeroed,
  // not expose the stale values that were there before the shrink.
  std::ranges::fill(v, 3.0);
  v.resize(2);
  v.resize(10);
  CHECK(v(0) == 3.0);
  CHECK(v(1) == 3.0);
  for (index_type i = 2; i < 10; ++i) {
    CHECK(v(i) == 0.0);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Uninitialized construction of user types
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("uninitialized_vector", "[uninitialized][vector]") {
  SECTION("static") {
    Vector<double, 3> v(uninitialized);
    v = Vector<double, 3>{1, 2, 3};
    CHECK(v == Vector<double, 3>{1, 2, 3});

    Vector<double, 3> v2(uninitialized, 3);
    CHECK(v2.size() == 3);
  }
  SECTION("dynamic") {
    VectorX<double> v(uninitialized, 100);
    CHECK(v.size() == 100);
    std::iota(v.begin(), v.end(), 0.0);
    CHECK(v(0) == 0.0);
    CHECK(v(99) == 99.0);
  }
  SECTION("dynamic_from_extents") {
    VectorX<double> v(uninitialized, extents<dynamic_extent>(5));
    CHECK(v.size() == 5);
  }
}

TEST_CASE("uninitialized_matrix", "[uninitialized][matrix]") {
  SECTION("static") {
    Matrix<double, 2, 3> m(uninitialized);
    m = Matrix<double, 2, 3>{{1, 2, 3}, {4, 5, 6}};
    CHECK(m(1, 2) == 6.0);

    Matrix<double, 2, 3> m2(uninitialized, 2, 3);
    CHECK(m2.rows() == 2);
    CHECK(m2.cols() == 3);
  }
  SECTION("fully_dynamic") {
    MatrixXX<double> m(uninitialized, 4, 5);
    CHECK(m.rows() == 4);
    CHECK(m.cols() == 5);
    for (index_type i = 0; i < 4; ++i) {
      for (index_type j = 0; j < 5; ++j) {
        m(i, j) = double(i * 5 + j);
      }
    }
    CHECK(m(3, 4) == 19.0);
  }
  SECTION("mixed_extents") {
    Matrix<double, dynamic_extent, 3> m(uninitialized, 7);
    CHECK(m.rows() == 7);
    CHECK(m.cols() == 3);
    Matrix<double, dynamic_extent, 3> m2(uninitialized, 7, 3);
    CHECK(m2.rows() == 7);
  }
  SECTION("from_extents") {
    MatrixXX<double> m(uninitialized, create_dextents(2, 6));
    CHECK(m.rows() == 2);
    CHECK(m.cols() == 6);
  }
}

TEST_CASE("uninitialized_array_tensor_form_dataarray",
          "[uninitialized][array][tensor][form][data_array]") {
  Array<double, 3, 3> a(uninitialized);
  a(2, 2) = 1.0;
  CHECK(a(2, 2) == 1.0);
  Array<double, dynamic_extent> ax(uninitialized, 6);
  CHECK(ax.extent(0) == 6);

  Tensor<double, 2, 2, 2> t(uninitialized);
  t(1, 1, 1) = 5.0;
  CHECK(t(1, 1, 1) == 5.0);
  Tensor<double, dynamic_extent, 2> tx(uninitialized, 5);
  CHECK(tx.extent(0) == 5);
  CHECK(tx.extent(1) == 2);

  Form<double, 3> f(uninitialized);
  f(0) = 1.0;
  CHECK(f(0) == 1.0);
  Form<double, dynamic_extent> fx(uninitialized, 4);
  CHECK(fx.extent(0) == 4);

  DataArray<double, 3> d(uninitialized);
  d(1) = 2.0;
  CHECK(d(1) == 2.0);
  DataArray<double, dynamic_extent> dx(uninitialized, 4);
  CHECK(dx.extent(0) == 4);
}

TEST_CASE("uninitialized_quaternion", "[uninitialized][quaternion]") {
  Quaternion<double> q(uninitialized);
  q(0) = 1.0;
  q(1) = 0.0;
  q(2) = 0.0;
  q(3) = 0.0;
  CHECK(q == Quaternion<double>::identity());
}

// ─────────────────────────────────────────────────────────────────────────────
// Uninitialized resize
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("uninitialized_resize_preserves_prefix", "[uninitialized][resize]") {
  SECTION("vector") {
    VectorX<double> v(uninitialized, 4);
    std::iota(v.begin(), v.end(), 1.0);
    v.resize(uninitialized, 10);
    CHECK(v.size() == 10);
    for (index_type i = 0; i < 4; ++i) {
      CHECK(v(i) == double(i + 1));
    }
    v.resize(uninitialized, 2);
    CHECK(v.size() == 2);
    CHECK(v(1) == 2.0);
  }
  SECTION("matrix_rank_dynamic_1") {
    Matrix<double, dynamic_extent, 2> m(uninitialized, 3);
    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m.resize(uninitialized, 6);
    CHECK(m.rows() == 6);
    // row-major: the first row is the linear prefix
    CHECK(m(0, 0) == 1.0);
    CHECK(m(0, 1) == 2.0);
  }
  SECTION("matrix_fully_dynamic") {
    MatrixXX<double> m(uninitialized, 2, 2);
    m.resize(uninitialized, 8, 9);
    CHECK(m.rows() == 8);
    CHECK(m.cols() == 9);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dynamic DenseData value semantics
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("dense_data_dynamic_value_semantics", "[uninitialized][storage]") {
  using D = storage::DenseData<double, std::dynamic_extent>;

  SECTION("default_is_empty") {
    D d;
    CHECK(d.size() == 0);
    CHECK(d.capacity() == 0);
    CHECK(d.begin() == d.end());
  }
  SECTION("zero_size") {
    D d(0);
    CHECK(d.size() == 0);
    D u(uninitialized, 0);
    CHECK(u.size() == 0);
    D c = u;
    CHECK(c.size() == 0);
  }
  SECTION("copy_is_deep") {
    D a(uninitialized, 5);
    std::iota(a.begin(), a.end(), 0.0);
    D b = a;
    CHECK(b.size() == 5);
    CHECK(b.data() != a.data());
    CHECK(std::ranges::equal(a, b));
    b.coeff_ref(0) = 100.0;
    CHECK(a.coeff(0) == 0.0);
  }
  SECTION("copy_assign_reuses_capacity") {
    D a(8);
    const auto *buf = a.data();
    D b(uninitialized, 3);
    std::iota(b.begin(), b.end(), 1.0);
    a = b;
    CHECK(a.data() == buf);
    CHECK(a.size() == 3);
    CHECK(a.capacity() == 8);
    CHECK(std::ranges::equal(a, b));
  }
  SECTION("copy_assign_grows") {
    D a(2);
    D b(uninitialized, 9);
    std::iota(b.begin(), b.end(), 1.0);
    a = b;
    CHECK(a.size() == 9);
    CHECK(std::ranges::equal(a, b));
  }
  SECTION("self_assignment") {
    D a(uninitialized, 4);
    std::iota(a.begin(), a.end(), 1.0);
    auto &ref = a;
    a = ref;
    CHECK(a.size() == 4);
    CHECK(a.coeff(3) == 4.0);
  }
  SECTION("move_leaves_source_empty") {
    D a(uninitialized, 6);
    std::iota(a.begin(), a.end(), 1.0);
    const auto *buf = a.data();
    D b = std::move(a);
    CHECK(b.data() == buf);
    CHECK(b.size() == 6);
    CHECK(a.size() == 0);     // NOLINT(bugprone-use-after-move)
    CHECK(a.capacity() == 0); // NOLINT(bugprone-use-after-move)

    D c;
    c = std::move(b);
    CHECK(c.data() == buf);
    CHECK(b.size() == 0); // NOLINT(bugprone-use-after-move)
  }
  SECTION("shrink_keeps_buffer") {
    D a(10);
    const auto *buf = a.data();
    a.resize(3);
    CHECK(a.data() == buf);
    CHECK(a.capacity() == 10);
    a.resize(uninitialized, 10);
    CHECK(a.data() == buf);
    CHECK(a.size() == 10);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal use: paths that allocate uninitialized and fully overwrite
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("internal_construction_from_expression",
          "[uninitialized][internal]") {
  // ZipperBase converting constructor / eval() allocate without zero-fill
  // and rely on evaluation writing every coefficient.
  SECTION("dynamic") {
    VectorX<double> a(uninitialized, 1000);
    std::iota(a.begin(), a.end(), 0.0);
    VectorX<double> b = (2.0 * a).eval();
    CHECK(b.size() == 1000);
    for (index_type i = 0; i < 1000; ++i) {
      CHECK(b(i) == 2.0 * double(i));
    }
    VectorX<double> c(a + b);
    CHECK(c(999) == 3.0 * 999.0);
  }
  SECTION("static") {
    Matrix<double, 3, 3> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    Matrix<double, 3, 3> At = A.transpose();
    CHECK(At(0, 2) == 7.0);
    CHECK(At(2, 0) == 3.0);
    Matrix<double, 3, 3> AAt = (A * At).eval();
    CHECK(AAt(0, 0) == 14.0);
  }
  SECTION("non_trivial_value_type") {
    VectorX<std::string> s(3);
    s(0) = "a";
    s(1) = "b";
    s(2) = "c";
    VectorX<std::string> t(s.as_span());
    CHECK(t(0) == "a");
    CHECK(t(2) == "c");
  }
}

TEST_CASE("internal_span_make_owned", "[uninitialized][internal]") {
  VectorX<double> a(uninitialized, 16);
  std::iota(a.begin(), a.end(), 1.0);
  auto span = a.as_span();
  auto owned = span.expression().make_owned();
  CHECK(owned.extent(0) == 16);
  CHECK(owned(0) == 1.0);
  CHECK(owned(15) == 16.0);
  a(0) = -1.0;
  CHECK(owned(0) == 1.0); // independent storage
}

TEST_CASE("internal_assignment_resizes", "[uninitialized][internal]") {
  SECTION("grow_via_assignment") {
    VectorX<double> small(2);
    VectorX<double> big(uninitialized, 50);
    std::iota(big.begin(), big.end(), 0.0);
    small = big * 1.0;
    CHECK(small.size() == 50);
    CHECK(small(49) == 49.0);
  }
  SECTION("shrink_then_regrow_overwrites_stale_values") {
    VectorX<double> v(10);
    std::ranges::fill(v, 99.0);
    VectorX<double> two{1.0, 2.0};
    v = two;
    CHECK(v.size() == 2);
    VectorX<double> ten(uninitialized, 10);
    std::iota(ten.begin(), ten.end(), 0.0);
    v = ten + 0.0 * ten; // non-trivial expression -> snapshot path
    CHECK(v.size() == 10);
    for (index_type i = 0; i < 10; ++i) {
      CHECK(v(i) == double(i));
    }
  }
  SECTION("self_aliasing_snapshot") {
    MatrixXX<double> M{{1, 2}, {3, 4}};
    M = (M * M).eval();
    CHECK(M(0, 0) == 7.0);
    CHECK(M(1, 1) == 22.0);
    MatrixXX<double> N{{1, 2}, {3, 4}};
    N = N * N; // aliasing product: snapshot temporary is allocated uninitialized
    CHECK(N(0, 1) == 10.0);
    CHECK(N(1, 0) == 15.0);
  }
  SECTION("scalar_broadcast_fills_everything") {
    VectorX<double> v(uninitialized, 8);
    v = expression::nullary::Constant<double>(3.0);
    CHECK(all_zero(v) == false);
    CHECK(std::ranges::all_of(v, [](double x) { return x == 3.0; }));
  }
}

TEST_CASE("internal_initializer_lists_and_components",
          "[uninitialized][internal]") {
  Vector<double, 3> v{1, 2, 3};
  CHECK(v(2) == 3.0);
  CHECK_THROWS(Vector<double, 3>{1, 2});
  VectorX<double> vx{4, 5, 6, 7};
  CHECK(vx.size() == 4);
  CHECK(vx(3) == 7.0);
  MatrixXX<double> m{{1, 2, 3}, {4, 5, 6}};
  CHECK(m(1, 2) == 6.0);
  CHECK_THROWS(MatrixXX<double>{{1, 2, 3}, {4, 5}});
  Quaternion<double> q(1.0, 2.0, 3.0, 4.0);
  CHECK(q(0) == 1.0);
  CHECK(q(3) == 4.0);
  CHECK(all_zero(DataArray<double, 3>::zero()));
  CHECK(all_zero(DataArray<double, dynamic_extent>::zero(5)));
}
