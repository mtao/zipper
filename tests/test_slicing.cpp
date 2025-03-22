
#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <iostream>
#include <uvl/Matrix.hpp>
#include <uvl/Vector.hpp>
#include <uvl/VectorBase.hpp>
#include <uvl/detail/dynamic_extents_indices.hpp>
#include <uvl/detail/swizzle_extents.hpp>
#include <uvl/storage/PlainObjectStorage.hpp>
#include <uvl/types.hpp>
#include <uvl/views/nullary/ConstantView.hpp>
#include <uvl/views/nullary/RandomView.hpp>

namespace {
void print(uvl::concepts::ArrayBaseDerived auto const& M) {
    for (uvl::index_type j = 0; j < M.extent(0); ++j) {
        for (uvl::index_type k = 0; k < M.extent(1); ++k) {
            std::cout << M(j, k) << " ";
        }
        std::cout << std::endl;
    }
}

void print(uvl::concepts::MatrixBaseDerived auto const& M) {
    for (uvl::index_type j = 0; j < M.extent(0); ++j) {
        for (uvl::index_type k = 0; k < M.extent(1); ++k) {
            std::cout << M(j, k) << " ";
        }
        std::cout << std::endl;
    }
}
void print(uvl::concepts::VectorBaseDerived auto const& M) {
    for (uvl::index_type j = 0; j < M.extent(0); ++j) {
        std::cout << M(j) << " ";
    }
    std::cout << std::endl;
}
}  // namespace
   //
TEST_CASE("test_matrix_slice_shapes", "[extents][matrix][slice]") {
    spdlog::info("Manipulating MN: ");
    uvl::MatrixBase RN(uvl::views::nullary::normal_random_view<double>(
        uvl::extents<4, 5>{}, 0, 20));
    uvl::Matrix MN = RN;
    {
        auto full_slice = MN.slice<uvl::full_extent_t, uvl::full_extent_t>();

        using ST = std::decay_t<decltype(full_slice.view())>;
        static_assert(ST::actionable_indices.size() == 2);
        static_assert(ST::actionable_indices[0] == 0);
        static_assert(ST::actionable_indices[1] == 1);

        static_assert(std::is_same_v<ST::extents_type, uvl::extents<4, 5>>);

        REQUIRE(3 == full_slice.view().get_index<0>(3, 4));
        REQUIRE(4 == full_slice.view().get_index<1>(3, 4));
    }
    {
        auto slice = MN.slice<std::integral_constant<uvl::index_type, 1>,
                              uvl::full_extent_t>();
        using ST = std::decay_t<decltype(slice.view())>;
        using T = ST::slice_storage_type;

        static_assert(
            std::is_same_v<uvl::full_extent_type, std::tuple_element_t<1, T>>);

        static_assert(std::is_same_v<std::integral_constant<uvl::index_type, 1>,
                                     std::tuple_element_t<0, T>>);

        static_assert(uvl::concepts::SliceLike<std::tuple_element_t<0, T>>);
        static_assert(uvl::concepts::SliceLike<std::tuple_element_t<1, T>>);

        static_assert(uvl::concepts::IndexLike<std::tuple_element_t<0, T>>);
        static_assert(!uvl::concepts::IndexLike<std::tuple_element_t<1, T>>);

        static_assert(
            std::is_same_v<uvl::full_extent_type, std::tuple_element_t<1, T>>);
        static_assert(ST::actionable_indices.size() == 2);
        static_assert(ST::actionable_indices[0] == std::dynamic_extent);
        static_assert(ST::actionable_indices[1] == 0);
        static_assert(std::is_same_v<ST::extents_type, uvl::extents<5>>);

        REQUIRE(1 == slice.view().get_index<0>(4));
        REQUIRE(4 == slice.view().get_index<1>(4));
    }
    {
        auto slice = MN.slice<uvl::full_extent_type,
                              std::integral_constant<uvl::index_type, 1>>();
        using ST = std::decay_t<decltype(slice.view())>;
        using T = ST::slice_storage_type;

        static_assert(
            std::is_same_v<uvl::full_extent_type, std::tuple_element_t<0, T>>);

        static_assert(std::is_same_v<std::integral_constant<uvl::index_type, 1>,
                                     std::tuple_element_t<1, T>>);

        static_assert(uvl::concepts::SliceLike<std::tuple_element_t<0, T>>);
        static_assert(uvl::concepts::SliceLike<std::tuple_element_t<1, T>>);

        static_assert(uvl::concepts::IndexLike<std::tuple_element_t<1, T>>);
        static_assert(!uvl::concepts::IndexLike<std::tuple_element_t<0, T>>);

        static_assert(
            std::is_same_v<uvl::full_extent_type, std::tuple_element_t<0, T>>);
        static_assert(ST::actionable_indices.size() == 2);
        static_assert(ST::actionable_indices[1] == std::dynamic_extent);
        static_assert(ST::actionable_indices[0] == 0);
        static_assert(std::is_same_v<ST::extents_type, uvl::extents<4>>);

        REQUIRE(1 == slice.view().get_index<1>(4));
        REQUIRE(4 == slice.view().get_index<0>(4));
    }
}

TEST_CASE("test_matrix_slicing", "[extents][matrix][slice]") {
    spdlog::info("Manipulating MN: ");
    uvl::MatrixBase RN(uvl::views::nullary::normal_random_view<double>(
        uvl::extents<4, 4>{}, 0, 20));
    uvl::Matrix MN = RN;
    auto full_slice = MN.slice<uvl::full_extent_t, uvl::full_extent_t>();

    REQUIRE(MN.extents() == full_slice.extents());
    spdlog::info("MN");
    print(MN);
    spdlog::info("full slice");
    print(MN.slice<uvl::full_extent_t, uvl::full_extent_t>());
    spdlog::info("full slice");
    print(MN.slice<uvl::full_extent_t, uvl::full_extent_t>());
    spdlog::info("are same");
    print((MN.slice<uvl::full_extent_t, uvl::full_extent_t>().as_array() ==
           MN.as_array()));

    CHECK((MN.slice<uvl::full_extent_t, uvl::full_extent_t>() == MN));

    auto slice = MN.slice<std::integral_constant<uvl::index_type, 1>,
                          uvl::full_extent_t>();
    REQUIRE(slice.extent(0) == 4);
    REQUIRE(slice.extents().rank() == 1);
    CHECK(slice(0) == MN(1, 0));
    CHECK(slice(1) == MN(1, 1));
    CHECK(slice(2) == MN(1, 2));
    CHECK(slice(3) == MN(1, 3));
    slice(0) = 2.0;
    slice(1) = 100000;
    slice(2) = 104000;
    slice(3) = 100003;

    CHECK(2.0 == MN(1, 0));
    CHECK(100000 == MN(1, 1));
    CHECK(104000 == MN(1, 2));
    CHECK(100003 == MN(1, 3));

    slice = uvl::views::nullary::ConstantView<double>(3.4);
    CHECK(3.4 == MN(1, 0));
    CHECK(3.4 == MN(1, 1));
    CHECK(3.4 == MN(1, 2));
    CHECK(3.4 == MN(1, 3));

    MN.row<std::integral_constant<uvl::index_type, 2>>() = slice;

    CHECK(3.4 == MN(2, 0));
    CHECK(3.4 == MN(2, 1));
    CHECK(3.4 == MN(2, 2));
    CHECK(3.4 == MN(2, 3));
    MN.col<std::integral_constant<uvl::index_type, 2>>() = slice;

    CHECK(3.4 == MN(0, 2));
    CHECK(3.4 == MN(1, 2));
    CHECK(3.4 == MN(2, 2));
    CHECK(3.4 == MN(3, 2));
    MN.col(2) = slice * 2;

    CHECK(2 * 3.4 == MN(0, 2));
    CHECK(2 * 3.4 == MN(1, 2));
    CHECK(2 * 3.4 == MN(2, 2));
    CHECK(2 * 3.4 == MN(3, 2));

    MN = RN;

    CHECK(slice(0) == MN(1, 0));
    CHECK(slice(1) == MN(1, 1));
    CHECK(slice(2) == MN(1, 2));
    CHECK(slice(3) == MN(1, 3));
    return;
    auto diag = MN.diagonal();
    REQUIRE(diag.extents() == uvl::create_dextents(4));
    CHECK(diag(0) == MN(0, 0));
    CHECK(diag(1) == MN(1, 1));
    CHECK(diag(2) == MN(2, 2));
    CHECK(diag(3) == MN(3, 3));

    auto slice2 =
        MN(std::integral_constant<uvl::index_type, 3>{}, uvl::full_extent_t{});
    CHECK(slice2(0) == MN(3, 0));
    CHECK(slice2(1) == MN(3, 1));
    CHECK(slice2(2) == MN(3, 2));
    CHECK(slice2(3) == MN(3, 3));

    uvl::Matrix NN = MN;
    slice2 = 100 * slice;

    CHECK(slice2(0) == 100 * NN(1, 0));
    CHECK(slice2(1) == 100 * NN(1, 1));
    CHECK(slice2(2) == 100 * NN(1, 2));
    CHECK(slice2(3) == 100 * NN(1, 3));
    NN = MN;
    auto slice3 = MN(2, uvl::full_extent);
    slice3 = 50 * slice2;
    CHECK(slice3(0) == 50 * NN(1, 0));
    CHECK(slice3(1) == 50 * NN(1, 1));
    CHECK(slice3(2) == 50 * NN(1, 2));
    CHECK(slice3(3) == 50 * 100 * NN(1, 3));
}
