
#include <zipper/storage/DenseAccessor.hpp>
#include <zipper/storage/DynamicDenseData.hpp>
#include <zipper/storage/StaticDenseData.hpp>
#include <zipper/storage/concepts/AccessorLike.hpp>

#include "../catch_include.hpp"

using namespace zipper;
namespace {
void test(zipper::storage::concepts::AccessorLike auto& data) {
    REQUIRE(data.size() == 30);
    REQUIRE(data.extent(0) == 5);
    REQUIRE(data.extent(1) == 6);
    for (index_type i = 0; i < data.extent(0); ++i) {
        for (index_type j = 0; j < data.extent(1); ++j) {
            data.coeff_ref(i, j) = i * 10 + j;
        }
    }

    for (index_type i = 0; i < data.extent(0); ++i) {
        for (index_type j = 0; j < data.extent(1); ++j) {
            index_type index = i * 10 + j;
            CHECK(data.coeff_ref(i, j) == index);
            CHECK(data.const_coeff_ref(i, j) == index);
            CHECK(data.coeff(i, j) == index);
        }
    }
    for (index_type i = 0; i < data.extent(0); ++i) {
        for (index_type j = 0; j < data.extent(1); ++j) {
            data.coeff_ref(i, j) = j * 10 + i;
        }
    }
    for (index_type i = 0; i < data.extent(0); ++i) {
        for (index_type j = 0; j < data.extent(1); ++j) {
            index_type index = j * 10 + i;
            CHECK(data.coeff_ref(i, j) == index);
            CHECK(data.const_coeff_ref(i, j) == index);
            CHECK(data.coeff(i, j) == index);
        }
    }
}
}  // namespace
TEST_CASE("static_dense_accessor", "[data]") {
    zipper::storage::DenseAccessor<
        zipper::storage::StaticDenseData<index_type, 30>, extents<5, 6>,
        zipper::default_layout_policy, zipper::default_accessor_policy<int>>
        A;
    test(A);
    auto B = A.as_span();
    test(B);
    // test_static<5>(A);
    // test_dense<5>(A);
}
TEST_CASE("dynamic_dense_accessor", "[data]") {
    zipper::storage::DenseAccessor<
        zipper::storage::DynamicDenseData<index_type>, dextents<2>,
        zipper::default_layout_policy, zipper::default_accessor_policy<int>>
        A(extents(5, 6));
    test(A);
    auto B = A.as_span();
    test(B);
    // test_static<5>(A);
    // test_dense<5>(A);
}
