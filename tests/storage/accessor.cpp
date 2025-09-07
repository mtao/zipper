
#include <zipper/storage/DenseAccessor.hpp>
#include <zipper/storage/DynamicDenseData.hpp>
#include <zipper/storage/StaticDenseData.hpp>
#include <zipper/storage/concepts/AccessorLike.hpp>

#include "../catch_include.hpp"

using namespace zipper;
namespace {
void test(zipper::storage::concepts::AccessorLike auto& data) {
    REQUIRE(data.size() == 30);
    REQUIRE(data.extent.coeff(0) == 5);
    REQUIRE(data.extent(1) == 6);
    for (index_type i = 0; i < data.extent.coeff(0); ++i) {
        for (index_type j = 0; j < data.extent(1); ++j) {
            data.coeff_ref(i, j) = i * 10 + j;
        }
    }

    for (index_type i = 0; i < data.extent.coeff(0); ++i) {
        for (index_type j = 0; j < data.extent(1); ++j) {
            index_type index = i * 10 + j;
            CHECK(data.coeff_ref(i, j) == index);
            CHECK(data.const_coeff_ref(i, j) == index);
            CHECK(data.coeff(i, j) == index);
        }
    }
    for (index_type i = 0; i < data.extent.coeff(0); ++i) {
        for (index_type j = 0; j < data.extent(1); ++j) {
            data.coeff_ref(i, j) = j * 10 + i;
        }
    }
    for (index_type i = 0; i < data.extent.coeff(0); ++i) {
        for (index_type j = 0; j < data.extent(1); ++j) {
            index_type index = j * 10 + i;
            CHECK(data.coeff_ref(i, j) == index);
            CHECK(data.const_coeff_ref(i, j) == index);
            CHECK(data.coeff(i, j) == index);
        }
    }
}
}  // namespace

TEST_CASE("standard_data_are_data_like", "[storage]") {
    {
        using T = zipper::storage::StaticDenseData<index_type, 30>;
        T t;
        static_assert(std::is_same_v<index_type, std::decay_t<T>::value_type>);
        static_assert(
            std::is_same_v<index_type, std::decay_t<T>::element_type>);
        static_assert(std::is_same_v<decltype(t.coeff(0)), index_type>);
        static_assert(zipper::storage::concepts::DataLike<T>);
    }
    {
        using T = zipper::storage::DynamicDenseData<index_type>;
        T t(30);
        static_assert(std::is_same_v<index_type, std::decay_t<T>::value_type>);
        static_assert(
            std::is_same_v<index_type, std::decay_t<T>::element_type>);
        static_assert(std::is_same_v<decltype(t.coeff(0)), index_type>);
        static_assert(zipper::storage::concepts::DataLike<T>);
    }
    {
        using T = zipper::storage::SpanData<index_type, 5>;
        std::array<index_type, 5> a;
        T t = std::span(a);
        static_assert(std::is_same_v<index_type, std::decay_t<T>::value_type>);
        static_assert(
            std::is_same_v<index_type, std::decay_t<T>::element_type>);
        static_assert(std::is_same_v<decltype(t.coeff(0)), index_type>);
        static_assert(zipper::storage::concepts::DataLike<T>);
    }
    {
        using T = zipper::storage::SpanData<index_type, std::dynamic_extent>;
        std::vector<index_type> a(5);
        T t = std::span(a);
        static_assert(std::is_same_v<index_type, std::decay_t<T>::value_type>);
        static_assert(
            std::is_same_v<index_type, std::decay_t<T>::element_type>);
        static_assert(std::is_same_v<decltype(t.coeff(0)), index_type>);
        static_assert(zipper::storage::concepts::DataLike<T>);
    }

    {
        using T = zipper::storage::StaticDenseData<const index_type, 5>;
        T t = {{0, 1, 2, 3, 4}};
        static_assert(std::is_same_v<index_type, std::decay_t<T>::value_type>);
        static_assert(
            std::is_same_v<const index_type, std::decay_t<T>::element_type>);
        static_assert(std::is_same_v<decltype(t.coeff(0)), index_type>);
        static_assert(zipper::storage::concepts::DataLike<T>);
    }

    // if constexpr (false) {  // NOTE: std::vector cannot take non-const value
    //                         // types
    //     using T = zipper::storage::DynamicDenseData<const index_type>;
    //     T t = std::vector<const index_type>({0, 1, 2, 3, 4});
    //     static_assert(std::is_same_v<decltype(t.coeff(0)), index_type>);
    // }
    {
        using T = zipper::storage::SpanData<const index_type, 5>;
        const std::array<index_type, 5> a = {{0, 1, 2, 3, 4}};
        T t = std::span(a);
        static_assert(std::is_same_v<index_type, std::decay_t<T>::value_type>);
        static_assert(
            std::is_same_v<const index_type, std::decay_t<T>::element_type>);
        static_assert(std::is_same_v<decltype(t.coeff(0)), index_type>);
        static_assert(zipper::storage::concepts::DataLike<T>);
    }
    {
        using T =
            zipper::storage::SpanData<const index_type, std::dynamic_extent>;
        const std::vector<index_type> a(5);
        T t = std::span(a);
        static_assert(std::is_same_v<index_type, std::decay_t<T>::value_type>);
        static_assert(
            std::is_same_v<const index_type, std::decay_t<T>::element_type>);
        static_assert(std::is_same_v<decltype(t.coeff(0)), index_type>);
        static_assert(zipper::storage::concepts::DataLike<T>);
    }

    //===================================================

    {
        using T = const zipper::storage::StaticDenseData<index_type, 30>;
        T t = {{0, 1, 2, 3, 4}};
        static_assert(std::is_same_v<index_type, std::decay_t<T>::value_type>);
        static_assert(
            std::is_same_v<index_type, std::decay_t<T>::element_type>);
        static_assert(std::is_same_v<decltype(t.coeff(0)), index_type>);
    }
    {
        using T = const zipper::storage::DynamicDenseData<index_type>;
        T t(30);
        static_assert(std::is_same_v<index_type, std::decay_t<T>::value_type>);
        static_assert(
            std::is_same_v<index_type, std::decay_t<T>::element_type>);
        static_assert(std::is_same_v<decltype(t.coeff(0)), index_type>);
    }
    {
        using T = const zipper::storage::StaticDenseData<const index_type, 30>;
        T t = {{0, 1, 2, 3, 4}};
        static_assert(std::is_same_v<index_type, std::decay_t<T>::value_type>);
        static_assert(
            std::is_same_v<const index_type, std::decay_t<T>::element_type>);
        static_assert(std::is_same_v<decltype(t.coeff(0)), index_type>);
    }
    // if constexpr (false) {  // NOTE: std::vector cannot take non-const value
    //     using T = const zipper::storage::DynamicDenseData<const index_type>;
    //     T t(30);
    //     static_assert(std::is_same_v<decltype(t.coeff(0)), index_type>);
    // }
}

/*
TEST_CASE("static_dense_accessor", "[data]") {
    zipper::storage::DenseAccessor<
        zipper::storage::StaticDenseData<index_type, 30>, extents<5, 6>,
        zipper::default_layout_policy, zipper::default_accessor_policy<int>>
        A;
    test(A);
    auto B = A.as_span();
    test(B);
}
TEST_CASE("dynamic_dense_accessor", "[data]") {
    zipper::storage::DynamicDenseData<index_type> data(30);
    // old clang doesn't like this full line
    using T = zipper::storage::DenseAccessor<
        zipper::storage::DynamicDenseData<index_type>, dextents<2>,
        zipper::default_layout_policy, zipper::default_accessor_policy<int>>;
    T A(std::move(data), create_dextents(5, 6));

    // zipper::storage::DenseAccessor<
    //     zipper::storage::DynamicDenseData<index_type>, dextents<2>,
    //     zipper::default_layout_policy, zipper::default_accessor_policy<int>>
    //     A(extents(5, 6));
    test(A);
    auto B = A.as_span();
    test(B);
    // test_static<5>(A);
    // test_dense<5>(A);
}
TEST_CASE("dynamic_static_dense_accessor", "[data]") {
    zipper::storage::StaticDenseData<index_type, 30> data;
    zipper::storage::DenseAccessor<
        zipper::storage::StaticDenseData<index_type, 30>, dextents<2>,
        zipper::default_layout_policy, zipper::default_accessor_policy<int>>
        A(std::move(data), create_dextents(5, 6));

    test(A);
    auto B = A.as_span();
    test(B);
}
TEST_CASE("static_dynamic_dense_accessor", "[data]") {
    zipper::storage::DynamicDenseData<index_type> data(30);
    zipper::storage::DenseAccessor<
        zipper::storage::DynamicDenseData<index_type>, extents<5, 6>,
        zipper::default_layout_policy, zipper::default_accessor_policy<int>>
        A(std::move(data));

    test(A);
    auto B = A.as_span();
    test(B);
    zipper::storage::DenseAccessor<
        zipper::storage::DynamicDenseData<index_type>, extents<5, 6>,
        zipper::default_layout_policy, zipper::default_accessor_policy<int>>
        C;

    CHECK(C.size() == 30);
    CHECK(C.linear_access().size() == 30);
}
*/
