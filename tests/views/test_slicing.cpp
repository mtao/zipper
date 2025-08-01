

#include <iostream>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/VectorBase.hpp>
#include <zipper/detail/extents/dynamic_extents_indices.hpp>
#include <zipper/detail/extents/swizzle_extents.hpp>
#include <zipper/storage/PlainObjectStorage.hpp>
#include <zipper/types.hpp>
#include <zipper/views/nullary/ConstantView.hpp>
#include <zipper/views/nullary/RandomView.hpp>

#include "../catch_include.hpp"
#include "../fmt_include.hpp"

namespace {
void print(zipper::concepts::ArrayBaseDerived auto const& M) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
        for (zipper::index_type k = 0; k < M.extent(1); ++k) {
            std::cout << M(j, k) << " ";
        }
        std::cout << std::endl;
    }
}

void print(zipper::concepts::MatrixBaseDerived auto const& M) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
        for (zipper::index_type k = 0; k < M.extent(1); ++k) {
            std::cout << M(j, k) << " ";
        }
        std::cout << std::endl;
    }
}
void print(zipper::concepts::VectorBaseDerived auto const& M) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
        std::cout << M(j) << " ";
    }
    std::cout << std::endl;
}
}  // namespace
   //
TEST_CASE("test_matrix_slice_shapes", "[extents][matrix][slice]") {
    spdlog::info("Manipulating MN: ");
    zipper::MatrixBase RN(zipper::views::nullary::normal_random_view<double>(
        zipper::extents<4, 5>{}, 0, 20));
    zipper::Matrix MN = RN;
    {
        auto full_slice =
            MN.slice<zipper::full_extent_t, zipper::full_extent_t>();

        using ST = std::decay_t<decltype(full_slice.view())>;
        static_assert(ST::actionable_indices.size() == 2);
        static_assert(ST::actionable_indices[0] == 0);
        static_assert(ST::actionable_indices[1] == 1);

        static_assert(std::is_same_v<ST::extents_type, zipper::extents<4, 5>>);

        REQUIRE(3 == full_slice.view().get_index<0>(3, 4));
        REQUIRE(4 == full_slice.view().get_index<1>(3, 4));
    }
    {
        auto slice = MN.slice<std::integral_constant<zipper::index_type, 1>,
                              zipper::full_extent_t>();
        using ST = std::decay_t<decltype(slice.view())>;
        using T = ST::slice_storage_type;

        static_assert(std::is_same_v<zipper::full_extent_type,
                                     std::tuple_element_t<1, T>>);

        static_assert(
            std::is_same_v<std::integral_constant<zipper::index_type, 1>,
                           std::tuple_element_t<0, T>>);

        static_assert(zipper::concepts::SliceLike<std::tuple_element_t<0, T>>);
        static_assert(zipper::concepts::SliceLike<std::tuple_element_t<1, T>>);

        static_assert(zipper::concepts::IndexLike<std::tuple_element_t<0, T>>);
        static_assert(!zipper::concepts::IndexLike<std::tuple_element_t<1, T>>);

        static_assert(std::is_same_v<zipper::full_extent_type,
                                     std::tuple_element_t<1, T>>);
        static_assert(ST::actionable_indices.size() == 2);
        static_assert(ST::actionable_indices[0] == std::dynamic_extent);
        static_assert(ST::actionable_indices[1] == 0);
        static_assert(std::is_same_v<ST::extents_type, zipper::extents<5>>);

        REQUIRE(1 == slice.view().get_index<0>(4));
        REQUIRE(4 == slice.view().get_index<1>(4));
    }
    {
        auto slice = MN.slice<zipper::full_extent_type,
                              std::integral_constant<zipper::index_type, 1>>();
        using ST = std::decay_t<decltype(slice.view())>;
        using T = ST::slice_storage_type;

        static_assert(std::is_same_v<zipper::full_extent_type,
                                     std::tuple_element_t<0, T>>);

        static_assert(
            std::is_same_v<std::integral_constant<zipper::index_type, 1>,
                           std::tuple_element_t<1, T>>);

        static_assert(zipper::concepts::SliceLike<std::tuple_element_t<0, T>>);
        static_assert(zipper::concepts::SliceLike<std::tuple_element_t<1, T>>);

        static_assert(zipper::concepts::IndexLike<std::tuple_element_t<1, T>>);
        static_assert(!zipper::concepts::IndexLike<std::tuple_element_t<0, T>>);

        static_assert(std::is_same_v<zipper::full_extent_type,
                                     std::tuple_element_t<0, T>>);
        static_assert(ST::actionable_indices.size() == 2);
        static_assert(ST::actionable_indices[1] == std::dynamic_extent);
        static_assert(ST::actionable_indices[0] == 0);
        static_assert(std::is_same_v<ST::extents_type, zipper::extents<4>>);

        REQUIRE(1 == slice.view().get_index<1>(4));
        REQUIRE(4 == slice.view().get_index<0>(4));
    }
}

TEST_CASE("test_matrix_slicing", "[extents][matrix][slice]") {
    spdlog::info("Manipulating MN: ");
    zipper::MatrixBase RN(zipper::views::nullary::normal_random_view<double>(
        zipper::extents<4, 4>{}, 0, 20));

    zipper::Matrix MN = RN;
    zipper::Matrix<double, std::dynamic_extent, std::dynamic_extent> MND = MN;
    auto full_slice = MN.slice<zipper::full_extent_t, zipper::full_extent_t>();
    auto full_sliceD =
        MND.slice<zipper::full_extent_t, zipper::full_extent_t>();

    REQUIRE(MN.extents() == full_slice.extents());
    REQUIRE(full_sliceD.extents() == full_slice.extents());
    spdlog::info("MN");
    print(MN);
    spdlog::info("full slice");
    print(MN.slice<zipper::full_extent_t, zipper::full_extent_t>());
    spdlog::info("full slice");
    print(MN.slice<zipper::full_extent_t, zipper::full_extent_t>());
    spdlog::info("are same");
    print(
        (MN.slice<zipper::full_extent_t, zipper::full_extent_t>().as_array() ==
         MN.as_array()));

    CHECK((MN.slice<zipper::full_extent_t, zipper::full_extent_t>() == MN));
    CHECK((MN.slice(zipper::full_extent_t{}, zipper::full_extent_t{}) == MN));
    CHECK((MND.slice<zipper::full_extent_t, zipper::full_extent_t>() == MN));
    CHECK((MND.slice(zipper::full_extent_t{}, zipper::full_extent_t{}) == MN));

    // zipper::slice(zipper::index_type(1));
    {
        auto S = MN.slice(zipper::slice(1, MN.extent(0) - 1),
                          zipper::full_extent_t{});

        auto SD = MND.slice(zipper::slice(1, MND.extent(0) - 1),
                            zipper::full_extent_t{});

        REQUIRE(S.extent(0) == MN.extent(0) - 1);
        REQUIRE(S.extent(1) == MN.extent(1));

        REQUIRE(SD.extents() == S.extents());
        for (const auto& [a, b] :
             zipper::utils::extents::all_extents_indices(S.extents())) {
            static_assert(std::is_integral_v<std::decay_t<decltype(a)>>);
            static_assert(std::is_integral_v<std::decay_t<decltype(b)>>);
            CHECK(S(a, b) == MN(a + 1, b));
            CHECK(SD(a, b) == S(a, b));
        }
    }

    auto slice = MN.slice<std::integral_constant<zipper::index_type, 1>,
                          zipper::full_extent_t>();
    auto sliceD = MND.slice<std::integral_constant<zipper::index_type, 1>,
                            zipper::full_extent_t>();
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

    REQUIRE(sliceD.extent(0) == 4);
    REQUIRE(sliceD.extents().rank() == 1);
    CHECK(sliceD(0) == MND(1, 0));
    CHECK(sliceD(1) == MND(1, 1));
    CHECK(sliceD(2) == MND(1, 2));
    CHECK(sliceD(3) == MND(1, 3));
    sliceD(0) = 2.0;
    sliceD(1) = 100000;
    sliceD(2) = 104000;
    sliceD(3) = 100003;

    CHECK(2.0 == MN(1, 0));
    CHECK(100000 == MN(1, 1));
    CHECK(104000 == MN(1, 2));
    CHECK(100003 == MN(1, 3));
    CHECK(2.0 == MND(1, 0));
    CHECK(100000 == MND(1, 1));
    CHECK(104000 == MND(1, 2));
    CHECK(100003 == MND(1, 3));

    slice = zipper::views::nullary::ConstantView<double>(3.4);
    sliceD = zipper::views::nullary::ConstantView<double>(3.4);
    CHECK(3.4 == MN(1, 0));
    CHECK(3.4 == MN(1, 1));
    CHECK(3.4 == MN(1, 2));
    CHECK(3.4 == MN(1, 3));
    CHECK(3.4 == MND(1, 0));
    CHECK(3.4 == MND(1, 1));
    CHECK(3.4 == MND(1, 2));
    CHECK(3.4 == MND(1, 3));

    MN.row<std::integral_constant<zipper::index_type, 2>>() = slice;
    MND.row<std::integral_constant<zipper::index_type, 2>>() = slice;

    CHECK(3.4 == MN(2, 0));
    CHECK(3.4 == MN(2, 1));
    CHECK(3.4 == MN(2, 2));
    CHECK(3.4 == MN(2, 3));

    CHECK(3.4 == MND(2, 0));
    CHECK(3.4 == MND(2, 1));
    CHECK(3.4 == MND(2, 2));
    CHECK(3.4 == MND(2, 3));
    MN.col<std::integral_constant<zipper::index_type, 2>>() = slice;

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
    REQUIRE(diag.extents() == zipper::create_dextents(4));
    CHECK(diag(0) == MN(0, 0));
    CHECK(diag(1) == MN(1, 1));
    CHECK(diag(2) == MN(2, 2));
    CHECK(diag(3) == MN(3, 3));

    auto slice2 = MN(std::integral_constant<zipper::index_type, 3>{},
                     zipper::full_extent_t{});
    CHECK(slice2(0) == MN(3, 0));
    CHECK(slice2(1) == MN(3, 1));
    CHECK(slice2(2) == MN(3, 2));
    CHECK(slice2(3) == MN(3, 3));

    zipper::Matrix NN = MN;
    slice2 = 100 * slice;

    CHECK(slice2(0) == 100 * NN(1, 0));
    CHECK(slice2(1) == 100 * NN(1, 1));
    CHECK(slice2(2) == 100 * NN(1, 2));
    CHECK(slice2(3) == 100 * NN(1, 3));
    NN = MN;
    auto slice3 = MN(2, zipper::full_extent_t{});
    slice3 = 50 * slice2;
    CHECK(slice3(0) == 50 * NN(1, 0));
    CHECK(slice3(1) == 50 * NN(1, 1));
    CHECK(slice3(2) == 50 * NN(1, 2));
    CHECK(slice3(3) == 50 * 100 * NN(1, 3));

    auto slice4 = MN(3, zipper::full_extent_t{});
    CHECK(slice4(0) == MN(3, 0));
    CHECK(slice4(1) == MN(3, 1));
    CHECK(slice4(2) == MN(3, 2));
    CHECK(slice4(3) == MN(3, 3));
}

TEST_CASE("test_vector_slice_assignment", "[extents][vector][slice]") {
    zipper::Vector<double, 4> x;

    x(3) = 2;

    x.head<3>() = zipper::views::nullary::ConstantView<double>(1);
    CHECK(x(0) == 1);
    CHECK(x(1) == 1);
    CHECK(x(2) == 1);
    CHECK(x(3) == 2);
}

TEST_CASE("test_partial_slice", "[extents][tensor][slice]") {
    spdlog::info("Manipulating MN: ");
}

TEST_CASE("test_span_array_access", "[vector][storage][dense][span]") {
    {
        zipper::Vector<double, 3> x = {1, 2, 3};

        std::array<zipper::index_type, 2> a{{0, 2}};
        auto s = x(a);
        REQUIRE(s.extents().rank() == 2);
        CHECK(s(0) == 1);
        CHECK(s(1) == 3);
    }
    /*
    {
        zipper::Vector<double, std::dynamic_extent> x = {1, 2, 3};

        std::vector<index_type> a{{0,2}};
        auto s = x(a);
        REQUIRE(s.extents().rank() == 2);
        CHECK(s(0) == 1);
        CHECK(s(0) == 3);



    }
    */
}
