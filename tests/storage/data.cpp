#include "../catch_include.hpp"
#include <zipper/storage/DenseData.hpp>
#include <zipper/storage/SpanData.hpp>
#include <zipper/storage/concepts/Data.hpp>
#include <zipper/storage/concepts/DenseData.hpp>
#include <zipper/storage/concepts/DynamicData.hpp>
#include <zipper/storage/concepts/StaticData.hpp>

namespace {
void test(zipper::storage::concepts::Data auto &data, size_t N) {
  static_assert(
      std::is_same_v<typename std::decay_t<decltype(data)>::value_type,
                     double>);

  for (zipper::index_type j = 0; j < N; ++j) {
    data.coeff_ref(j) = double(j);
  }
  for (zipper::index_type j = 0; j < N; ++j) {
    CHECK(data.coeff(j) == double(j));
    CHECK(data.coeff_ref(j) == double(j));
    CHECK(data.const_coeff_ref(j) == double(j));
  }
  for (zipper::index_type j = 0; j < N; ++j) {
    data.coeff_ref(j) = double(0);
  }
  for (zipper::index_type j = 0; j < N; ++j) {
    CHECK(data.coeff(j) == 0);
    CHECK(data.coeff_ref(j) == 0);
    CHECK(data.const_coeff_ref(j) == 0);
  }
}
} // namespace
void test_dynamic(zipper::storage::concepts::DynamicData auto &data, size_t N) {
  static_assert(std::decay_t<decltype(data)>::static_size ==
                std::dynamic_extent);
  test(data, N);
  data.resize(20);
  test(data, 20);
}

template <int N, typename T>
  requires zipper::storage::concepts::StaticData<N, T>
void test_static(T &data) {
  constexpr static zipper::index_type size =
      std::decay_t<decltype(data)>::static_size;
  if constexpr (size == std::dynamic_extent) {
    assert(data.size() == N);
  } else {
    static_assert(size == N);
  }
  test(data, data.size());
}

template <size_t N>
void test_dense(zipper::storage::concepts::DenseData auto &data) {
  zipper::storage::SpanData<double, N> B(
      data.as_std_span().template subspan<0, N>());
  zipper::storage::SpanData<double, std::dynamic_extent> C(data.as_std_span());
  test_static<N>(B);
  test(C, N);
}

using namespace zipper;
TEST_CASE("static_dense_data", "[data]") {
  zipper::storage::DenseData<double, 5> A;
  test_static<5>(A);
  test_dense<5>(A);
}

TEST_CASE("dynamic_dense_data", "[data]") {
  {
    zipper::storage::DenseData<double, std::dynamic_extent> A(5);
    test_dense<5>(A);
    test_dynamic(A, 5);
    // dynamic resizes to 20
    test_dense<20>(A);
  }
}
