
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
    static_assert(std::is_same_v<T, typename InfoType::template get_type<0>>);
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
    static_assert(std::is_same_v<T, typename InfoType::template get_type<0>>);
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
    static_assert(std::is_same_v<T, typename InfoType::template get_type<1>>);
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
    static_assert(std::is_same_v<T, typename InfoType::template get_type<1>>);
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
