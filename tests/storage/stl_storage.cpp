
#include <zipper/expression/nullary/StlMDArray.hpp>
#include <zipper/expression/nullary/Random.hpp>
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

    // access_features should indicate writable
    static_assert(!traits::access_features.is_const);
    static_assert(traits::access_features.is_reference);

    // derived bools must agree
    static_assert(traits::is_writable);
    static_assert(zipper::expression::detail::get_is_coefficient_consistent<traits>());

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
    static_assert(zipper::expression::detail::get_is_coefficient_consistent<traits>());
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

TEST_CASE("stl_mdarray_assign_random_uniform_with_resize",
          "[storage][stl][assign]") {
  // Verify that a MatrixBase wrapping a non-owning StlMDArray over
  // std::vector<std::array<double, Dim+1>> can be assigned from a
  // uniform_random<double> expression when pts does not know the size it
  // is being assigned to. The assignment should resize the underlying
  // vector automatically.

  constexpr zipper::index_type Dim = 2;

  std::vector<std::array<double, Dim + 1>> pts;  // starts empty

  zipper::MatrixBase E =
      zipper::expression::nullary::get_non_owning_stl_storage(pts);

  // pts is empty, so E should have 0 rows
  REQUIRE(E.rows() == 0);
  REQUIRE(E.cols() == Dim + 1);

  // First assignment: resize from 0 → 5 rows
  {
    constexpr zipper::index_type num_rows = 5;
    E = zipper::expression::nullary::uniform_random<double>(
        zipper::extents<std::dynamic_extent, Dim + 1>{num_rows}, 10.0, 100.0);

    CHECK(E.rows() == num_rows);
    CHECK(E.cols() == Dim + 1);
    CHECK(pts.size() == num_rows);

    for (zipper::index_type r = 0; r < num_rows; ++r) {
      for (zipper::index_type c = 0; c < Dim + 1; ++c) {
        CHECK(pts[r][c] >= 10.0);
        CHECK(pts[r][c] < 100.0);
        CHECK(E(r, c) == pts[r][c]);
      }
    }
  }

  // Second assignment: resize from 5 → 3 rows (shrink)
  {
    constexpr zipper::index_type num_rows = 3;
    E = zipper::expression::nullary::uniform_random<double>(
        zipper::extents<std::dynamic_extent, Dim + 1>{num_rows}, 10.0, 100.0);

    CHECK(E.rows() == num_rows);
    CHECK(E.cols() == Dim + 1);
    CHECK(pts.size() == num_rows);

    for (zipper::index_type r = 0; r < num_rows; ++r) {
      for (zipper::index_type c = 0; c < Dim + 1; ++c) {
        CHECK(pts[r][c] >= 10.0);
        CHECK(pts[r][c] < 100.0);
        CHECK(E(r, c) == pts[r][c]);
      }
    }
  }

  // Third assignment: resize from 3 → 8 rows (grow)
  {
    constexpr zipper::index_type num_rows = 8;
    E = zipper::expression::nullary::uniform_random<double>(
        zipper::extents<std::dynamic_extent, Dim + 1>{num_rows}, 10.0, 100.0);

    CHECK(E.rows() == num_rows);
    CHECK(E.cols() == Dim + 1);
    CHECK(pts.size() == num_rows);

    for (zipper::index_type r = 0; r < num_rows; ++r) {
      for (zipper::index_type c = 0; c < Dim + 1; ++c) {
        CHECK(pts[r][c] >= 10.0);
        CHECK(pts[r][c] < 100.0);
        CHECK(E(r, c) == pts[r][c]);
      }
    }
  }
}

// TODO: test resizing vectors and how it affects the expression

#include <zipper/Vector.hpp>

TEST_CASE("stl_storage_unwrap_zipper_static_asserts", "[storage][stl][unwrap]") {
  using Vec3 = zipper::Vector<double, 3>;

  SECTION("NoUnwrap: array of Vec3 is rank 1 with value_type=Vec3") {
    using Info = zipper::storage::StlStorageInfo<std::array<Vec3, 4>>;
    static_assert(Info::rank == 1);
    static_assert(std::is_same_v<Info::value_type, Vec3>);
    static_assert(Info::extents_type::rank() == 1);
    static_assert(Info::extents_type::static_extent(0) == 4);
  }

  SECTION("UnwrapZipper: array of Vec3 is rank 2 with value_type=double") {
    using Info = zipper::storage::StlStorageInfo<std::array<Vec3, 4>,
                                                  zipper::storage::UnwrapZipper>;
    static_assert(Info::rank == 2);
    static_assert(std::is_same_v<Info::value_type, double>);
    static_assert(Info::extents_type::rank() == 2);
    // extents should be <4, 3>: 4 from array, 3 from Vec3
    static_assert(Info::extents_type::static_extent(0) == 4);
    static_assert(Info::extents_type::static_extent(1) == 3);
  }

  SECTION("UnwrapZipper: vector of Vec3 is rank 2, dynamic x 3") {
    using Info = zipper::storage::StlStorageInfo<std::vector<Vec3>,
                                                  zipper::storage::UnwrapZipper>;
    static_assert(Info::rank == 2);
    static_assert(std::is_same_v<Info::value_type, double>);
    static_assert(Info::extents_type::rank() == 2);
    static_assert(Info::extents_type::static_extent(0) == std::dynamic_extent);
    static_assert(Info::extents_type::static_extent(1) == 3);
  }

  SECTION("UnwrapZipper: array of Matrix is rank 3") {
    using Mat23 = zipper::Matrix<double, 2, 3>;
    using Info = zipper::storage::StlStorageInfo<std::array<Mat23, 5>,
                                                  zipper::storage::UnwrapZipper>;
    static_assert(Info::rank == 3);
    static_assert(std::is_same_v<Info::value_type, double>);
    static_assert(Info::extents_type::rank() == 3);
    // extents should be <5, 2, 3>: 5 from array, 2 rows and 3 cols from Matrix
    static_assert(Info::extents_type::static_extent(0) == 5);
    static_assert(Info::extents_type::static_extent(1) == 2);
    static_assert(Info::extents_type::static_extent(2) == 3);
  }

  SECTION("HasZipperInterface concept checks") {
    static_assert(zipper::storage::detail::HasZipperInterface<Vec3>);
    static_assert(zipper::storage::detail::HasZipperInterface<zipper::Matrix<double, 2, 3>>);
    static_assert(!zipper::storage::detail::HasZipperInterface<double>);
    static_assert(!zipper::storage::detail::HasZipperInterface<int>);
    static_assert(!zipper::storage::detail::HasZipperInterface<std::array<double, 3>>);
  }
}

TEST_CASE("stl_storage_unwrap_zipper_coefficient_access",
          "[storage][stl][unwrap]") {
  using Vec3 = zipper::Vector<double, 3>;

  SECTION("Read/write through UnwrapZipper non-owning view of array<Vec3,N>") {
    std::array<Vec3, 2> data;
    data[0] = Vec3({1.0, 2.0, 3.0});
    data[1] = Vec3({4.0, 5.0, 6.0});

    auto view = zipper::expression::nullary::get_non_owning_stl_storage<
        zipper::storage::UnwrapZipper>(data);

    auto e = view.extents();
    CHECK(e.extent(0) == 2);
    CHECK(e.extent(1) == 3);

    // Read coefficients: view(row, col) should match data[row](col)
    CHECK(view.coeff(0, 0) == 1.0);
    CHECK(view.coeff(0, 1) == 2.0);
    CHECK(view.coeff(0, 2) == 3.0);
    CHECK(view.coeff(1, 0) == 4.0);
    CHECK(view.coeff(1, 1) == 5.0);
    CHECK(view.coeff(1, 2) == 6.0);

    // Write through the view and verify it modifies the originals
    view.coeff_ref(0, 0) = 10.0;
    view.coeff_ref(0, 1) = 20.0;
    view.coeff_ref(0, 2) = 30.0;
    view.coeff_ref(1, 0) = 40.0;
    view.coeff_ref(1, 1) = 50.0;
    view.coeff_ref(1, 2) = 60.0;

    // Verify via the view
    CHECK(view.const_coeff_ref(0, 0) == 10.0);
    CHECK(view.const_coeff_ref(0, 1) == 20.0);
    CHECK(view.const_coeff_ref(0, 2) == 30.0);
    CHECK(view.const_coeff_ref(1, 0) == 40.0);
    CHECK(view.const_coeff_ref(1, 1) == 50.0);
    CHECK(view.const_coeff_ref(1, 2) == 60.0);

    // Verify the original data was modified
    CHECK(data[0](0) == 10.0);
    CHECK(data[0](1) == 20.0);
    CHECK(data[0](2) == 30.0);
    CHECK(data[1](0) == 40.0);
    CHECK(data[1](1) == 50.0);
    CHECK(data[1](2) == 60.0);
  }

  SECTION("Read/write through UnwrapZipper non-owning view of vector<Vec3>") {
    std::vector<Vec3> data;
    data.push_back(Vec3({7.0, 8.0, 9.0}));
    data.push_back(Vec3({10.0, 11.0, 12.0}));
    data.push_back(Vec3({13.0, 14.0, 15.0}));

    auto view = zipper::expression::nullary::get_non_owning_stl_storage<
        zipper::storage::UnwrapZipper>(data);

    auto e = view.extents();
    CHECK(e.extent(0) == 3);
    CHECK(e.extent(1) == 3);

    CHECK(view.coeff(0, 0) == 7.0);
    CHECK(view.coeff(0, 1) == 8.0);
    CHECK(view.coeff(0, 2) == 9.0);
    CHECK(view.coeff(1, 0) == 10.0);
    CHECK(view.coeff(2, 2) == 15.0);

    // Write and verify
    view.coeff_ref(2, 0) = 99.0;
    CHECK(data[2](0) == 99.0);
  }
}
