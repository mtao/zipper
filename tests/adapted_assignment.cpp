#include "catch_include.hpp"

#include <zipper/Array.hpp>
#include <zipper/COOMatrix.hpp>
#include <zipper/COOVector.hpp>
#include <zipper/CSMatrix.hpp>
#include <zipper/CSVector.hpp>
#include <zipper/DataArray.hpp>
#include <zipper/Form.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Quaternion.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/assignment_source.hpp>

#include <array>
#include <deque>
#include <list>
#include <memory_resource>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {
struct CustomBuffer { std::array<double, 3> values; };
struct UnregisteredBuffer { std::array<double, 3> values; };
struct InvalidAdapter {};

struct ScaledAccessor {
    using element_type = const double;
    using reference = double;
    using data_handle_type = const double *;
    using offset_policy = ScaledAccessor;
    double scale = 1;
    auto access(const double *pointer, size_t index) const -> double {
        return scale * pointer[index];
    }
    auto offset(const double *pointer, size_t index) const -> const double * {
        return pointer + index;
    }
};

template <typename Destination, typename Source>
concept CanAssign = requires(Destination &destination, Source &&source) {
    destination = std::forward<Source>(source);
};
} // namespace

namespace zipper {
template <>
struct assignment_source_traits<CustomBuffer> {
    static auto get(const CustomBuffer &source) {
        // A user can return a semantic wrapper, not just a raw expression.
        return VectorBase(std::span<const double, 3>(source.values));
    }
};
template <>
struct assignment_source_traits<InvalidAdapter> {
    static auto get(const InvalidAdapter &) -> int { return 1; }
};
} // namespace zipper

using namespace zipper;
using MappedVector = decltype(VectorBase(std::declval<std::array<double, 3> &>()));
using ConstMappedVector = decltype(VectorBase(std::declval<const std::array<double, 3> &>()));
using MappedMatrix = decltype(MatrixBase(
    std::declval<std::array<std::array<double, 3>, 2> &>()));

static_assert(detail::AdaptedAssignmentSource<std::array<double, 3>>);
static_assert(detail::AdaptedAssignmentSource<std::vector<double>>);
static_assert(detail::AdaptedAssignmentSource<std::vector<std::array<float, 3>>>);
static_assert(detail::AdaptedAssignmentSource<CustomBuffer>);
static_assert(!detail::AdaptedAssignmentSource<UnregisteredBuffer>);
static_assert(!detail::AdaptedAssignmentSource<InvalidAdapter>);
static_assert(!detail::AdaptedAssignmentSource<std::string>);
static_assert(!detail::AdaptedAssignmentSource<std::list<double>>);
static_assert(!detail::AdaptedAssignmentSource<std::deque<double>>);
static_assert(!detail::AdaptedAssignmentSource<std::vector<bool>>);
static_assert(!detail::AdaptedAssignmentSource<std::vector<std::vector<double>>>);
static_assert(!detail::AdaptedAssignmentSource<std::array<std::vector<double>, 3>>);
static_assert(!detail::AdaptedAssignmentSource<double *>);
static_assert(!detail::AdaptedAssignmentSource<double[3]>);
static_assert(!detail::AdaptedAssignmentSource<double>);
static_assert(!detail::AdaptedAssignmentSource<std::span<volatile double>>);
static_assert(!detail::AdaptedAssignmentSource<Vector<double, 3>>);
static_assert(!detail::AdaptedAssignmentSource<expression::nullary::MDArray<double, extents<3>>>);
static_assert(CanAssign<MappedVector, std::array<float, 3>>);
static_assert(!CanAssign<MappedVector, std::array<double, 2>>);
static_assert(!CanAssign<ConstMappedVector, std::array<double, 3>>);
static_assert(!CanAssign<MappedMatrix, std::array<double, 3>>);
static_assert(!CanAssign<MappedVector, std::array<std::array<double, 3>, 2>>);
static_assert(!CanAssign<MappedVector, UnregisteredBuffer>);
static_assert(!CanAssign<MappedVector, InvalidAdapter>);

TEST_CASE("adapted_assignment_owning_and_mapped_vectors", "[assignment][adapted]") {
    const std::array<int, 3> source{1, -2, 3};
    Vector<double, 3> fixed;
    fixed = source;
    CHECK(fixed(0) == 1);
    CHECK(fixed(1) == -2);
    CHECK(fixed(2) == 3);

    VectorX<double> dynamic(1);
    dynamic = source;
    REQUIRE(dynamic.size() == 3);
    CHECK(dynamic(1) == -2);
    dynamic = std::vector<float>{4, 5};
    REQUIRE(dynamic.size() == 2);
    CHECK(dynamic(0) == 4);
    CHECK(dynamic(1) == 5);

    std::array<double, 3> output{};
    auto mapped = VectorBase(output);
    mapped = source;
    CHECK(output == std::array<double, 3>{1, -2, 3});
    mapped = std::array<double, 3>{7, 8, 9};
    CHECK(output == std::array<double, 3>{7, 8, 9});
    mapped = output;
    CHECK(output == std::array<double, 3>{7, 8, 9});

    std::pmr::vector<double> allocated{10, 11, 12};
    mapped = allocated;
    CHECK(output == std::array<double, 3>{10, 11, 12});
}

TEST_CASE("adapted_assignment_rows_columns_and_slices", "[assignment][adapted]") {
    Matrix<double, 3, 3> matrix;
    matrix.row(1) = std::array<int, 3>{1, 2, 3};
    matrix.col(2) = std::vector<double>{4, 5, 6};
    CHECK(matrix(1, 0) == 1);
    CHECK(matrix(1, 1) == 2);
    CHECK(matrix(0, 2) == 4);
    CHECK(matrix(1, 2) == 5);
    CHECK(matrix(2, 2) == 6);
    matrix.col(0).head<2>() = std::array<double, 2>{8, 9};
    CHECK(matrix(0, 0) == 8);
    CHECK(matrix(1, 0) == 9);

    const std::array<std::array<int, 2>, 2> block{{{1, 2}, {3, 4}}};
    matrix.topRows<2>().leftCols<2>() = block;
    CHECK(matrix(0, 0) == 1);
    CHECK(matrix(0, 1) == 2);
    CHECK(matrix(1, 0) == 3);
    CHECK(matrix(1, 1) == 4);
}

TEST_CASE("adapted_assignment_rectangular_and_empty_containers", "[assignment][adapted]") {
    const std::array<std::array<int, 3>, 2> source{{{1, 2, 3}, {4, 5, 6}}};
    Matrix<double, dynamic_extent, dynamic_extent> matrix(1, 1);
    matrix = source;
    REQUIRE(matrix.rows() == 2);
    REQUIRE(matrix.cols() == 3);
    CHECK(matrix(1, 2) == 6);
    matrix = std::vector<std::array<float, 3>>{{7, 8, 9}};
    REQUIRE(matrix.rows() == 1);
    CHECK(matrix(0, 2) == 9);
    matrix = std::vector<std::array<float, 3>>{};
    CHECK(matrix.rows() == 0);
    CHECK(matrix.cols() == 3);
    VectorX<double> vector(4);
    vector = std::array<double, 0>{};
    CHECK(vector.size() == 0);
    vector = std::vector<double>{};
    CHECK(vector.size() == 0);
}

TEST_CASE("adapted_assignment_other_wrapper_families", "[assignment][adapted]") {
    const std::array<double, 3> source{2, 4, 6};
    Array<double, 3> array;
    array = source;
    CHECK(array(2) == 6);
    Form<double, 3> form;
    form = source;
    CHECK(form(1) == 4);
    DataArray<double, 3> data;
    data = source;
    CHECK(data(0) == 2);
    Tensor<double, 3> tensor;
    tensor = source;
    CHECK(tensor(2) == 6);
    Quaternion<double> quaternion;
    quaternion = std::array<double, 4>{1, 2, 3, 4};
    CHECK(quaternion.w() == 1);
    CHECK(quaternion.z() == 4);

    std::array<double, 3> destination{};
    auto array_view = ArrayBase(destination);
    array_view = source;
    CHECK(destination == source);
    auto form_view = FormBase(destination);
    form_view = std::array<double, 3>{3, 6, 9};
    CHECK(destination[2] == 9);
    auto data_view = DataArrayBase(destination);
    data_view = source;
    CHECK(destination[2] == 6);
    auto tensor_view = TensorBase(destination);
    tensor_view = std::array<double, 3>{4, 8, 12};
    CHECK(destination[2] == 12);

    using Cube = std::array<std::array<std::array<int, 2>, 2>, 2>;
    const Cube cube{{{{{1, 2}, {3, 4}}}, {{{5, 6}, {7, 8}}}}};
    Tensor<double, 2, 2, 2> rank_three;
    rank_three = cube;
    CHECK(rank_three(1, 1, 1) == 8);
}

TEST_CASE("adapted_assignment_mdspan_layout_stride_accessor", "[assignment][adapted]") {
    const std::array<double, 6> storage{1, 2, 3, 4, 5, 6};
    mdspan<const double, extents<2, 3>, storage::layout_left> column_major(storage.data());
    Matrix<double, 2, 3> matrix;
    matrix = column_major;
    CHECK(matrix(0, 0) == 1);
    CHECK(matrix(1, 0) == 2);
    CHECK(matrix(0, 2) == 5);
    CHECK(matrix(1, 2) == 6);

    using Strided = mdspan<const double, extents<3>, storage::layout_stride>;
    Strided strided(storage.data(), Strided::mapping_type(extents<3>{}, std::array<size_t, 1>{2}));
    Vector<double, 3> vector;
    vector = strided;
    CHECK(vector(0) == 1);
    CHECK(vector(1) == 3);
    CHECK(vector(2) == 5);
    using Scaled = mdspan<const double, extents<3>, storage::layout_right, ScaledAccessor>;
    Scaled scaled(storage.data(), Scaled::mapping_type{}, ScaledAccessor{2});
    vector = scaled;
    CHECK(vector(0) == 2);
    CHECK(vector(1) == 4);
    CHECK(vector(2) == 6);
}

TEST_CASE("adapted_assignment_aliasing_and_resize", "[assignment][adapted]") {
    std::array<double, 6> data{1, 2, 3, 4, 5, 6};
    auto destination = VectorBase(std::span<double>(data).subspan(1, 4));
    destination = std::span<const double>(data.data(), 4);
    CHECK(data == std::array<double, 6>{1, 1, 2, 3, 4, 6});

    VectorX<double> owner{1, 2, 3, 4, 5};
    const auto source = std::span<const double>(owner.expression().data() + 1, 3);
    owner = source;
    REQUIRE(owner.size() == 3);
    CHECK(owner(0) == 2);
    CHECK(owner(1) == 3);
    CHECK(owner(2) == 4);
}

TEST_CASE("adapted_assignment_extension_and_native_overloads", "[assignment][adapted]") {
    Vector<double, 3> vector;
    vector = CustomBuffer{{1, 3, 5}};
    CHECK(vector(2) == 5);
    std::array<double, 3> output{};
    auto mapped = VectorBase(output);
    mapped = CustomBuffer{{2, 4, 6}};
    CHECK(output == std::array<double, 3>{2, 4, 6});
    mapped = {7.0, 8.0, 9.0};
    CHECK(output[1] == 8);
    vector = vector + vector;
    CHECK(vector(2) == 10);

    VectorX<double> source{1, 2, 3};
    auto *pointer = source.expression().data();
    VectorX<double> moved;
    moved = std::move(source);
    CHECK(moved.expression().data() == pointer);
    VectorX<double> copied;
    copied = moved;
    CHECK(copied.expression().data() != moved.expression().data());
    CHECK(copied(2) == 3);
}

TEST_CASE("adapted_assignment_preserves_runtime_extent_checks", "[assignment][adapted]") {
    Vector<double, 3> vector{1, 2, 3};
    CHECK_THROWS_AS((vector = std::vector<double>{4, 5}), std::runtime_error);
    CHECK(vector(0) == 1);
    CHECK(vector(2) == 3);
    std::array<double, 3> output{1, 2, 3};
    auto mapped = VectorBase(output);
    CHECK_THROWS_AS((mapped = std::vector<double>{4, 5}), std::runtime_error);
    CHECK(output == std::array<double, 3>{1, 2, 3});
}

TEST_CASE("adapted_assignment_sparse_destinations", "[assignment][adapted]") {
    const std::array<double, 3> source{1, 0, 3};
    COOVector<double, 3> coordinate;
    coordinate = source;
    CHECK(coordinate(0) == 1);
    CHECK(coordinate(1) == 0);
    CHECK(coordinate(2) == 3);
    CSVector<double, 3> compressed;
    compressed = source;
    CHECK(compressed(0) == 1);
    CHECK(compressed(2) == 3);
    const std::array<std::array<double, 2>, 2> values{{{1, 0}, {0, 2}}};
    COOMatrix<double, 2, 2> coo;
    coo = values;
    CHECK(coo(1, 1) == 2);
    CSMatrix<double, 2, 2> cs;
    cs = values;
    CHECK(cs(0, 0) == 1);
    CHECK(cs(1, 1) == 2);
}
