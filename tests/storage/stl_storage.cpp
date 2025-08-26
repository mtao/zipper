// #include <zipper/MatrixBase.hpp>
// #include <zipper/VectorBase.hpp>

#include <zipper/storage/StlStorage.hpp>

#include "../catch_include.hpp"

TEST_CASE("stl_storage_basic_types", "[data]") {
    {
        using T = double;
        using InfoType = zipper::storage::detail::StlStorageInfo<T>;
        static_assert(InfoType::rank == 0);
        static_assert(std::is_same_v<double, InfoType::value_type>);
    }

    {
        using T = std::vector<double>;
        using InfoType = zipper::storage::detail::StlStorageInfo<T>;
        static_assert(InfoType::rank == 1);
        static_assert(std::is_same_v<double, InfoType::value_type>);
        static_assert(
            std::is_same_v<T, typename InfoType::template get_type<0>>);
        static_assert(InfoType::static_extent<0>() == zipper::dynamic_extent);

        std::vector<double> x = {0, 1, 2};
        zipper::storage::StlStorage storage(x);
        auto e = storage.extents();
        static_assert(e.static_extent(0) == std::dynamic_extent);
        CHECK(e.extent(0) == 3);

        CHECK(storage.coeff(0) == 0);
        CHECK(storage.coeff(1) == 1);
        CHECK(storage.coeff(2) == 2);
        CHECK(storage.coeff_ref(0) == 0);
        CHECK(storage.coeff_ref(1) == 1);
        CHECK(storage.coeff_ref(2) == 2);
        CHECK(storage.const_coeff_ref(0) == 0);
        CHECK(storage.const_coeff_ref(1) == 1);
        CHECK(storage.const_coeff_ref(2) == 2);

        storage.coeff_ref(0) = 4;
        storage.coeff_ref(1) = 5;
        storage.coeff_ref(2) = 6;
        CHECK(storage.const_coeff_ref(0) == 4);
        CHECK(storage.const_coeff_ref(1) == 5);
        CHECK(storage.const_coeff_ref(2) == 6);
    }

    {
        using T = std::array<double, 5>;
        using InfoType = zipper::storage::detail::StlStorageInfo<T>;
        static_assert(InfoType::rank == 1);
        static_assert(std::is_same_v<double, InfoType::value_type>);
        static_assert(
            std::is_same_v<T, typename InfoType::template get_type<0>>);
        static_assert(InfoType::static_extent<0>() == 5);

        std::array<double, 3> x{{0, 1, 2}};
        zipper::storage::StlStorage storage(x);
        auto e = storage.extents();
        static_assert(e.static_extent(0) == 3);
        CHECK(e.extent(0) == 3);

        CHECK(storage.coeff(0) == 0);
        CHECK(storage.coeff(1) == 1);
        CHECK(storage.coeff(2) == 2);
        CHECK(storage.coeff_ref(0) == 0);
        CHECK(storage.coeff_ref(1) == 1);
        CHECK(storage.coeff_ref(2) == 2);
        CHECK(storage.const_coeff_ref(0) == 0);
        CHECK(storage.const_coeff_ref(1) == 1);
        CHECK(storage.const_coeff_ref(2) == 2);

        storage.coeff_ref(0) = 4;
        storage.coeff_ref(1) = 5;
        storage.coeff_ref(2) = 6;
        CHECK(storage.const_coeff_ref(0) == 4);
        CHECK(storage.const_coeff_ref(1) == 5);
        CHECK(storage.const_coeff_ref(2) == 6);
    }
    {
        using T = std::vector<std::array<double, 5>>;
        using InfoType = zipper::storage::detail::StlStorageInfo<T>;
        static_assert(InfoType::rank == 2);
        static_assert(std::is_same_v<double, InfoType::value_type>);

        static_assert(std::is_same_v<std::array<double, 5>,
                                     typename InfoType::template get_type<0>>);
        static_assert(
            std::is_same_v<T, typename InfoType::template get_type<1>>);
        static_assert(InfoType::static_extent<0>() == 5);
        static_assert(InfoType::static_extent<1>() == zipper::dynamic_extent);

        std::vector<std::array<double, 3>> x{{0, 1, 2}, {3, 4, 5}};

        zipper::storage::StlStorage storage(x);
        auto e = storage.extents();
        static_assert(decltype(e)::static_extent(0) == std::dynamic_extent);
        static_assert(decltype(e)::static_extent(1) == 3);
        CHECK(e.extent(0) == 2);
        CHECK(e.extent(1) == 3);

        CHECK(storage.coeff(0, 0) == 0);
        CHECK(storage.coeff(0, 1) == 1);
        CHECK(storage.coeff(0, 2) == 2);
        CHECK(storage.coeff_ref(0, 0) == 0);
        CHECK(storage.coeff_ref(0, 1) == 1);
        CHECK(storage.coeff_ref(0, 2) == 2);
        CHECK(storage.const_coeff_ref(0, 0) == 0);
        CHECK(storage.const_coeff_ref(0, 1) == 1);
        CHECK(storage.const_coeff_ref(0, 2) == 2);

        CHECK(storage.coeff(1, 0) == 3);
        CHECK(storage.coeff(1, 1) == 4);
        CHECK(storage.coeff(1, 2) == 5);
        CHECK(storage.coeff_ref(1, 0) == 3);
        CHECK(storage.coeff_ref(1, 1) == 4);
        CHECK(storage.coeff_ref(1, 2) == 5);
        CHECK(storage.const_coeff_ref(1, 0) == 3);
        CHECK(storage.const_coeff_ref(1, 1) == 4);
        CHECK(storage.const_coeff_ref(1, 2) == 5);

        storage.coeff_ref(0, 0) = 14;
        storage.coeff_ref(0, 1) = 15;
        storage.coeff_ref(0, 2) = 16;
        storage.coeff_ref(1, 0) = 24;
        storage.coeff_ref(1, 1) = 25;
        storage.coeff_ref(1, 2) = 26;
        CHECK(storage.const_coeff_ref(0, 0) == 14);
        CHECK(storage.const_coeff_ref(0, 1) == 15);
        CHECK(storage.const_coeff_ref(0, 2) == 16);
        CHECK(storage.const_coeff_ref(1, 0) == 24);
        CHECK(storage.const_coeff_ref(1, 1) == 25);
        CHECK(storage.const_coeff_ref(1, 2) == 26);
    }

    {
        using T = std::array<std::vector<double>, 5>;
        using InfoType = zipper::storage::detail::StlStorageInfo<T>;
        static_assert(InfoType::rank == 2);
        static_assert(std::is_same_v<double, InfoType::value_type>);

        static_assert(std::is_same_v<std::vector<double>,
                                     typename InfoType::template get_type<0>>);
        static_assert(
            std::is_same_v<T, typename InfoType::template get_type<1>>);
        static_assert(InfoType::static_extent<0>() == zipper::dynamic_extent);
        static_assert(InfoType::static_extent<1>() == 5);

        std::array<std::vector<double>, 2> x{{{0, 1, 2}, {3, 4, 5}}};

        zipper::storage::StlStorage storage(x);
        auto e = storage.extents();
        static_assert(decltype(e)::static_extent(0) == 2);
        static_assert(decltype(e)::static_extent(1) == std::dynamic_extent);
        CHECK(e.extent(0) == 2);
        CHECK(e.extent(1) == 3);

        CHECK(storage.coeff(0, 0) == 0);
        CHECK(storage.coeff(0, 1) == 1);
        CHECK(storage.coeff(0, 2) == 2);
        CHECK(storage.coeff_ref(0, 0) == 0);
        CHECK(storage.coeff_ref(0, 1) == 1);
        CHECK(storage.coeff_ref(0, 2) == 2);
        CHECK(storage.const_coeff_ref(0, 0) == 0);
        CHECK(storage.const_coeff_ref(0, 1) == 1);
        CHECK(storage.const_coeff_ref(0, 2) == 2);

        CHECK(storage.coeff(1, 0) == 3);
        CHECK(storage.coeff(1, 1) == 4);
        CHECK(storage.coeff(1, 2) == 5);
        CHECK(storage.coeff_ref(1, 0) == 3);
        CHECK(storage.coeff_ref(1, 1) == 4);
        CHECK(storage.coeff_ref(1, 2) == 5);
        CHECK(storage.const_coeff_ref(1, 0) == 3);
        CHECK(storage.const_coeff_ref(1, 1) == 4);
        CHECK(storage.const_coeff_ref(1, 2) == 5);

        storage.coeff_ref(0, 0) = 14;
        storage.coeff_ref(0, 1) = 15;
        storage.coeff_ref(0, 2) = 16;
        storage.coeff_ref(1, 0) = 24;
        storage.coeff_ref(1, 1) = 25;
        storage.coeff_ref(1, 2) = 26;
        CHECK(storage.const_coeff_ref(0, 0) == 14);
        CHECK(storage.const_coeff_ref(0, 1) == 15);
        CHECK(storage.const_coeff_ref(0, 2) == 16);
        CHECK(storage.const_coeff_ref(1, 0) == 24);
        CHECK(storage.const_coeff_ref(1, 1) == 25);
        CHECK(storage.const_coeff_ref(1, 2) == 26);
    }
}
TEST_CASE("stl_storage_non_owning", "[data]") {
    {
        std::vector<double> x = {0, 1, 2};
        auto storage = zipper::storage::get_non_owning_stl_storage(x);
        auto e = storage.extents();
        static_assert(e.static_extent(0) == std::dynamic_extent);
        CHECK(e.extent(0) == 3);

        CHECK(storage.coeff(0) == 0);
        CHECK(storage.coeff(1) == 1);
        CHECK(storage.coeff(2) == 2);
        CHECK(storage.coeff_ref(0) == 0);
        CHECK(storage.coeff_ref(1) == 1);
        CHECK(storage.coeff_ref(2) == 2);
        CHECK(storage.const_coeff_ref(0) == 0);
        CHECK(storage.const_coeff_ref(1) == 1);
        CHECK(storage.const_coeff_ref(2) == 2);

        storage.coeff_ref(0) = 4;
        storage.coeff_ref(1) = 5;
        storage.coeff_ref(2) = 6;
        CHECK(storage.const_coeff_ref(0) == 4);
        CHECK(storage.const_coeff_ref(1) == 5);
        CHECK(storage.const_coeff_ref(2) == 6);
        CHECK(x[0] == 4);
        CHECK(x[1] == 5);
        CHECK(x[2] == 6);
    }

    {
        std::array<double, 3> x{{0, 1, 2}};
        auto storage = zipper::storage::get_non_owning_stl_storage(x);
        auto e = storage.extents();
        static_assert(e.static_extent(0) == 3);
        CHECK(e.extent(0) == 3);

        CHECK(storage.coeff(0) == 0);
        CHECK(storage.coeff(1) == 1);
        CHECK(storage.coeff(2) == 2);
        CHECK(storage.coeff_ref(0) == 0);
        CHECK(storage.coeff_ref(1) == 1);
        CHECK(storage.coeff_ref(2) == 2);
        CHECK(storage.const_coeff_ref(0) == 0);
        CHECK(storage.const_coeff_ref(1) == 1);
        CHECK(storage.const_coeff_ref(2) == 2);

        storage.coeff_ref(0) = 4;
        storage.coeff_ref(1) = 5;
        storage.coeff_ref(2) = 6;
        CHECK(storage.const_coeff_ref(0) == 4);
        CHECK(storage.const_coeff_ref(1) == 5);
        CHECK(storage.const_coeff_ref(2) == 6);

        CHECK(x[0] == 4);
        CHECK(x[1] == 5);
        CHECK(x[2] == 6);
    }
    {
        std::vector<std::array<double, 3>> x{{0, 1, 2}, {3, 4, 5}};

        auto storage = zipper::storage::get_non_owning_stl_storage(x);
        auto e = storage.extents();
        static_assert(decltype(e)::static_extent(0) == std::dynamic_extent);
        static_assert(decltype(e)::static_extent(1) == 3);
        CHECK(e.extent(0) == 2);
        CHECK(e.extent(1) == 3);

        CHECK(storage.coeff(0, 0) == 0);
        CHECK(storage.coeff(0, 1) == 1);
        CHECK(storage.coeff(0, 2) == 2);
        CHECK(storage.coeff_ref(0, 0) == 0);
        CHECK(storage.coeff_ref(0, 1) == 1);
        CHECK(storage.coeff_ref(0, 2) == 2);
        CHECK(storage.const_coeff_ref(0, 0) == 0);
        CHECK(storage.const_coeff_ref(0, 1) == 1);
        CHECK(storage.const_coeff_ref(0, 2) == 2);

        CHECK(storage.coeff(1, 0) == 3);
        CHECK(storage.coeff(1, 1) == 4);
        CHECK(storage.coeff(1, 2) == 5);
        CHECK(storage.coeff_ref(1, 0) == 3);
        CHECK(storage.coeff_ref(1, 1) == 4);
        CHECK(storage.coeff_ref(1, 2) == 5);
        CHECK(storage.const_coeff_ref(1, 0) == 3);
        CHECK(storage.const_coeff_ref(1, 1) == 4);
        CHECK(storage.const_coeff_ref(1, 2) == 5);

        storage.coeff_ref(0, 0) = 14;
        storage.coeff_ref(0, 1) = 15;
        storage.coeff_ref(0, 2) = 16;
        storage.coeff_ref(1, 0) = 24;
        storage.coeff_ref(1, 1) = 25;
        storage.coeff_ref(1, 2) = 26;
        CHECK(storage.const_coeff_ref(0, 0) == 14);
        CHECK(storage.const_coeff_ref(0, 1) == 15);
        CHECK(storage.const_coeff_ref(0, 2) == 16);
        CHECK(storage.const_coeff_ref(1, 0) == 24);
        CHECK(storage.const_coeff_ref(1, 1) == 25);
        CHECK(storage.const_coeff_ref(1, 2) == 26);

        CHECK(x[0][0] == 14);
        CHECK(x[0][1] == 15);
        CHECK(x[0][2] == 16);
        CHECK(x[1][0] == 24);
        CHECK(x[1][1] == 25);
        CHECK(x[1][2] == 26);
    }

    {
        std::array<std::vector<double>, 2> x{{{0, 1, 2}, {3, 4, 5}}};

        auto storage = zipper::storage::get_non_owning_stl_storage(x);
        auto e = storage.extents();
        static_assert(decltype(e)::static_extent(0) == 2);
        static_assert(decltype(e)::static_extent(1) == std::dynamic_extent);
        CHECK(e.extent(0) == 2);
        CHECK(e.extent(1) == 3);

        CHECK(storage.coeff(0, 0) == 0);
        CHECK(storage.coeff(0, 1) == 1);
        CHECK(storage.coeff(0, 2) == 2);
        CHECK(storage.coeff_ref(0, 0) == 0);
        CHECK(storage.coeff_ref(0, 1) == 1);
        CHECK(storage.coeff_ref(0, 2) == 2);
        CHECK(storage.const_coeff_ref(0, 0) == 0);
        CHECK(storage.const_coeff_ref(0, 1) == 1);
        CHECK(storage.const_coeff_ref(0, 2) == 2);

        CHECK(storage.coeff(1, 0) == 3);
        CHECK(storage.coeff(1, 1) == 4);
        CHECK(storage.coeff(1, 2) == 5);
        CHECK(storage.coeff_ref(1, 0) == 3);
        CHECK(storage.coeff_ref(1, 1) == 4);
        CHECK(storage.coeff_ref(1, 2) == 5);
        CHECK(storage.const_coeff_ref(1, 0) == 3);
        CHECK(storage.const_coeff_ref(1, 1) == 4);
        CHECK(storage.const_coeff_ref(1, 2) == 5);

        storage.coeff_ref(0, 0) = 14;
        storage.coeff_ref(0, 1) = 15;
        storage.coeff_ref(0, 2) = 16;
        storage.coeff_ref(1, 0) = 24;
        storage.coeff_ref(1, 1) = 25;
        storage.coeff_ref(1, 2) = 26;
        CHECK(storage.const_coeff_ref(0, 0) == 14);
        CHECK(storage.const_coeff_ref(0, 1) == 15);
        CHECK(storage.const_coeff_ref(0, 2) == 16);
        CHECK(storage.const_coeff_ref(1, 0) == 24);
        CHECK(storage.const_coeff_ref(1, 1) == 25);
        CHECK(storage.const_coeff_ref(1, 2) == 26);

        CHECK(x[0][0] == 14);
        CHECK(x[0][1] == 15);
        CHECK(x[0][2] == 16);
        CHECK(x[1][0] == 24);
        CHECK(x[1][1] == 25);
        CHECK(x[1][2] == 26);
    }
}

#include <zipper/Matrix.hpp>
#include <zipper/VectorBase.hpp>

TEST_CASE("stl_storage_zipper_bases", "[data]") {
    {
        std::vector<double> x = {0, 1, 2};
        auto storage = zipper::storage::get_non_owning_stl_storage(x);
        zipper::VectorBase X = storage;
        CHECK(X(0) == 0);
        CHECK(X(1) == 1);
        CHECK(X(2) == 2);

        CHECK(X.view().coeff(0) == 0);
        CHECK(X.view().coeff(1) == 1);
        CHECK(X.view().coeff(2) == 2);
        CHECK(X.view().coeff_ref(0) == 0);
        CHECK(X.view().coeff_ref(1) == 1);
        CHECK(X.view().coeff_ref(2) == 2);
        CHECK(X.view().const_coeff_ref(0) == 0);
        CHECK(X.view().const_coeff_ref(1) == 1);
        CHECK(X.view().const_coeff_ref(2) == 2);

        X(0) = 4;
        X(1) = 5;
        X(2) = 6;
        CHECK(X(0) == 4);
        CHECK(X(1) == 5);
        CHECK(X(2) == 6);

        using T = std::decay_t<decltype(X)>::view_type;
        using traits = zipper::views::detail::ViewTraits<T>;

        static_assert(!traits::is_const);
        static_assert(traits::is_writable);
        X = {0, 4, 5, 6};
        CHECK(x.size() == 4);
        CHECK(X.size() == 4);
        CHECK(X.rows() == 4);

        CHECK(X(0) == 0);
        CHECK(X(1) == 4);
        CHECK(X(2) == 5);
        CHECK(X(3) == 6);
    }

    {
        std::array<double, 3> x{{0, 1, 2}};
        auto storage = zipper::storage::get_non_owning_stl_storage(x);
        zipper::VectorBase X = storage;
        CHECK(X(0) == 0);
        CHECK(X(1) == 1);
        CHECK(X(2) == 2);

        CHECK(X.view().coeff(0) == 0);
        CHECK(X.view().coeff(1) == 1);
        CHECK(X.view().coeff(2) == 2);
        CHECK(X.view().coeff_ref(0) == 0);
        CHECK(X.view().coeff_ref(1) == 1);
        CHECK(X.view().coeff_ref(2) == 2);
        CHECK(X.view().const_coeff_ref(0) == 0);
        CHECK(X.view().const_coeff_ref(1) == 1);
        CHECK(X.view().const_coeff_ref(2) == 2);

        X(0) = 4;
        X(1) = 5;
        X(2) = 6;

        CHECK(X(0) == 4);
        CHECK(X(1) == 5);
        CHECK(X(2) == 6);

        X = {0, 4, 5};
        CHECK(x.size() == 3);
        CHECK(X.size() == 3);
        CHECK(X.rows() == 3);

        CHECK(X(0) == 0);
        CHECK(X(1) == 4);
        CHECK(X(2) == 5);
    }
    {
        std::vector<std::array<double, 3>> x{{0, 1, 2}, {3, 4, 5}};

        auto storage = zipper::storage::get_non_owning_stl_storage(x);
        zipper::MatrixBase X = storage;
        CHECK(X(0, 0) == 0);
        CHECK(X(0, 1) == 1);
        CHECK(X(0, 2) == 2);
        CHECK(X(1, 0) == 3);
        CHECK(X(1, 1) == 4);
        CHECK(X(1, 2) == 5);

        CHECK(X.view().coeff(0, 0) == 0);
        CHECK(X.view().coeff(0, 1) == 1);
        CHECK(X.view().coeff(0, 2) == 2);
        CHECK(X.view().coeff(1, 0) == 3);
        CHECK(X.view().coeff(1, 1) == 4);
        CHECK(X.view().coeff(1, 2) == 5);
        CHECK(X.view().coeff_ref(0, 0) == 0);
        CHECK(X.view().coeff_ref(0, 1) == 1);
        CHECK(X.view().coeff_ref(0, 2) == 2);
        CHECK(X.view().coeff_ref(1, 0) == 3);
        CHECK(X.view().coeff_ref(1, 1) == 4);
        CHECK(X.view().coeff_ref(1, 2) == 5);
        CHECK(X.view().const_coeff_ref(0, 0) == 0);
        CHECK(X.view().const_coeff_ref(0, 1) == 1);
        CHECK(X.view().const_coeff_ref(0, 2) == 2);
        CHECK(X.view().const_coeff_ref(1, 0) == 3);
        CHECK(X.view().const_coeff_ref(1, 1) == 4);
        CHECK(X.view().const_coeff_ref(1, 2) == 5);

        X(0, 0) = 14;
        X(0, 1) = 15;
        X(0, 2) = 16;
        X(1, 0) = 24;
        X(1, 1) = 25;
        X(1, 2) = 26;

        CHECK(x[0][0] == 14);
        CHECK(x[0][1] == 15);
        CHECK(x[0][2] == 16);
        CHECK(x[1][0] == 24);
        CHECK(x[1][1] == 25);
        CHECK(x[1][2] == 26);

        CHECK(X(0, 0) == 14);
        CHECK(X(0, 1) == 15);
        CHECK(X(0, 2) == 16);
        CHECK(X(1, 0) == 24);
        CHECK(X(1, 1) == 25);
        CHECK(X(1, 2) == 26);

        zipper::Matrix<double, 2, 3> B;
        CHECK(B.extents() == zipper::create_dextents(2, 3));
        CHECK(X.extents() == zipper::create_dextents(2, 3));
        B.row(0) = {0, 1, 2};
        B.row(1) = {3, 4, 5};
        X = B;
        CHECK(X.extents() == zipper::create_dextents(2, 3));
        CHECK((X == B));
        zipper::Matrix<double, 3, 3> C;
        CHECK(C.extents() == zipper::create_dextents(3, 3));
        C.row(0) = {0, 1, 2};
        C.row(1) = {3, 4, 5};
        C.row(2) = {6, 7, 8};
        X = C;
        CHECK(X.extents() == zipper::create_dextents(3, 3));
        CHECK((X == C));
    }

    {
        std::array<std::vector<double>, 2> x{{{0, 1, 2}, {3, 4, 5}}};

        auto storage = zipper::storage::get_non_owning_stl_storage(x);
        zipper::MatrixBase X = storage;
        CHECK(X(0, 0) == 0);
        CHECK(X(0, 1) == 1);
        CHECK(X(0, 2) == 2);
        CHECK(X(1, 0) == 3);
        CHECK(X(1, 1) == 4);
        CHECK(X(1, 2) == 5);

        X(0, 0) = 14;
        X(0, 1) = 15;
        X(0, 2) = 16;
        X(1, 0) = 24;
        X(1, 1) = 25;
        X(1, 2) = 26;

        CHECK(x[0][0] == 14);
        CHECK(x[0][1] == 15);
        CHECK(x[0][2] == 16);
        CHECK(x[1][0] == 24);
        CHECK(x[1][1] == 25);
        CHECK(x[1][2] == 26);

        CHECK(X(0, 0) == 14);
        CHECK(X(0, 1) == 15);
        CHECK(X(0, 2) == 16);
        CHECK(X(1, 0) == 24);
        CHECK(X(1, 1) == 25);
        CHECK(X(1, 2) == 26);
        zipper::Matrix<double, 2, 3> B;
        CHECK(B.extents() == zipper::create_dextents(2, 3));
        CHECK(X.extents() == zipper::create_dextents(2, 3));
        B.row(0) = {0, 1, 2};
        B.row(1) = {3, 4, 5};
        X = B;
        CHECK(X.extents() == zipper::create_dextents(2, 3));
        CHECK((X == B));
        zipper::Matrix<double, 2, 4> C;
        CHECK(C.extents() == zipper::create_dextents(2, 4));
        C.row(0) = {0, 1, 2, 8};
        C.row(1) = {3, 4, 5, 8};
        X = C;
        CHECK(X.extents() == zipper::create_dextents(2, 4));
        CHECK((X == C));
    }
}

