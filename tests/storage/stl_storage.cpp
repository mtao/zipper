
#include <zipper/expression/nullary/StlMDArray.hpp>
#include <zipper/storage/StlStorageInfo.hpp>
#include <zipper/Matrix.hpp>

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
    zipper::expression::nullary::StlMDArray storage(x);
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
    zipper::expression::nullary::StlMDArray storage(x);
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

    zipper::expression::nullary::StlMDArray storage(x);
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

    zipper::expression::nullary::StlMDArray storage(x);
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
    auto storage = zipper::expression::nullary::get_non_owning_stl_storage(x);
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
    auto storage = zipper::expression::nullary::get_non_owning_stl_storage(x);
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

    auto storage = zipper::expression::nullary::get_non_owning_stl_storage(x);
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

    auto storage = zipper::expression::nullary::get_non_owning_stl_storage(x);
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

TEST_CASE("stl_mdarray_traits_consistency", "[storage][stl]") {
  // Verify that ExpressionTraits for StlMDArray are self-consistent:
  // access_features/shape_features agree with is_writable, is_resizable, etc.

  // Non-const owning dynamic vector
  {
    using T = zipper::expression::nullary::StlMDArray<std::vector<double>>;
    using traits = zipper::expression::detail::ExpressionTraits<T>;

    // access_features should indicate writable, alias-free
    static_assert(!traits::access_features.is_const);
    static_assert(traits::access_features.is_reference);
    static_assert(traits::access_features.is_alias_free);

    // derived bools must agree
    static_assert(traits::is_writable);
    static_assert(traits::is_coefficient_consistent);

    // dynamic extent → resizable
    static_assert(traits::shape_features.is_resizable);
    static_assert(traits::is_resizable());
  }

  // Const non-owning reference
  {
    using T = zipper::expression::nullary::StlMDArray<const std::vector<double> &>;
    using traits = zipper::expression::detail::ExpressionTraits<T>;

    static_assert(traits::access_features.is_const);
    static_assert(!traits::is_writable);
    static_assert(!traits::is_assignable());
  }

  // Static extent → not resizable
  {
    using T = zipper::expression::nullary::StlMDArray<std::array<double, 3>>;
    using traits = zipper::expression::detail::ExpressionTraits<T>;

    static_assert(!traits::shape_features.is_resizable);
    static_assert(!traits::is_resizable());
  }

  // Dynamic 2D (vector<array>) → resizable
  {
    using T = zipper::expression::nullary::StlMDArray<
        std::vector<std::array<double, 3>>>;
    using traits = zipper::expression::detail::ExpressionTraits<T>;

    static_assert(traits::shape_features.is_resizable);
    static_assert(traits::is_resizable());
    static_assert(traits::is_writable);
    static_assert(traits::is_coefficient_consistent);
  }
}

TEST_CASE("stl_mdarray_transpose_assign", "[storage][stl]") {
  // Reproduces the bug from MshLoader: assigning a transposed MDSpan
  // into a non-owning StlMDArray matrix via MatrixBase::operator=.

  // Build a 2x3 source matrix via MDArray
  zipper::Matrix<double, 2, 3> src;
  src(0, 0) = 1;
  src(0, 1) = 2;
  src(0, 2) = 3;
  src(1, 0) = 4;
  src(1, 1) = 5;
  src(1, 2) = 6;

  // Destination: non-owning StlMDArray wrapping vector<array<double,2>>
  // After transpose, a 2×3 matrix becomes 3×2
  std::vector<std::array<double, 2>> dst_data(3, {0, 0});

  auto dst_storage =
      zipper::expression::nullary::get_non_owning_stl_storage(dst_data);
  zipper::MatrixBase dst(dst_storage);

  // Assign transpose of src (3x2) into dst (3x2)
  dst = src.transpose();

  CHECK(dst(0, 0) == 1);
  CHECK(dst(0, 1) == 4);
  CHECK(dst(1, 0) == 2);
  CHECK(dst(1, 1) == 5);
  CHECK(dst(2, 0) == 3);
  CHECK(dst(2, 1) == 6);

  // Also verify the underlying STL data was written through
  CHECK(dst_data[0][0] == 1);
  CHECK(dst_data[0][1] == 4);
  CHECK(dst_data[1][0] == 2);
  CHECK(dst_data[1][1] == 5);
  CHECK(dst_data[2][0] == 3);
  CHECK(dst_data[2][1] == 6);
}

// TODO: test resizing vectors and how it affects the expression
