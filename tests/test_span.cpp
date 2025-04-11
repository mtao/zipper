#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Form.hpp>
#include <zipper/Vector.hpp>
#include <zipper/views/nullary/ConstantView.hpp>
#include <zipper/views/nullary/IdentityView.hpp>
#include <zipper/views/nullary/RandomView.hpp>
#include <zipper/views/unary/PartialTraceView.hpp>
#include <zipper/views/unary/SwizzleView.hpp>

TEST_CASE("test_span_construction", "[vector][storage][dense]") {
    {
        std::array<double, 3> data;
        zipper::VectorSpan<double, 3> v(data);

        CHECK(&data[0] == &v(0));
        CHECK(&data[1] == &v(1));
        CHECK(&data[2] == &v(2));
    }
    {
        std::array<double, 3> data;
        zipper::VectorSpan v(data);
        zipper::TensorSpan<double,3> T(data);
        zipper::FormSpan<double,3> F(data);

        REQUIRE(v.extent(0) == data.size());
        static_assert(std::decay_t<decltype(v.extents())>::static_extent(0) == data.size());
        CHECK(&data[0] == &v(0));
        CHECK(&data[1] == &v(1));
        CHECK(&data[2] == &v(2));

        CHECK(&data[0] == &F(0));
        CHECK(&data[1] == &F(1));
        CHECK(&data[2] == &F(2));

        CHECK(&data[0] == &T(0));
        CHECK(&data[1] == &T(1));
        CHECK(&data[2] == &T(2));
    }

    {
        std::vector<double> data(20);

        zipper::VectorSpan v(data);
        zipper::TensorSpan<double,std::dynamic_extent> T(data);
        zipper::FormSpan<double,std::dynamic_extent> F(data);
        REQUIRE(v.extent(0) == data.size());
        for (size_t j = 0; j < data.size(); ++j) {
            CHECK(&data[j] == &v(j));
            CHECK(&data[j] == &T(j));
            CHECK(&data[j] == &F(j));
        }
    }
    {
        std::array<double, 20> data;
        zipper::MatrixSpan<double, 4, 5> M(data);
        zipper::TensorSpan<double,4,5> T(data);
        zipper::FormSpan<double,4,5> F(data);
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 5; ++k) {
                CHECK(&data[j * 5 + k] == &M(j, k));
                CHECK(&data[j * 5 + k] == &T(j, k));
                CHECK(&data[j * 5 + k] == &F(j, k));
            }
        }
    }

    {
        std::vector<double> data(20);

        using et = zipper::extents<4,5>;
        zipper::MatrixSpan<double,4,5, false> M(
            std::span<double, 20>(data.data(), data.data() + 20));
        zipper::TensorSpan_<double, et, false> T(
            std::span<double, 20>(data.data(), data.data() + 20));
        zipper::FormSpan_<double, et, false> F(
            std::span<double, 20>(data.data(), data.data() + 20));
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 5; ++k) {
                CHECK(&data[j * 5 + k] == &M(j, k));
            }
        }
    }
    {
        std::vector<double> data(20);

        using et = zipper::extents<4,5>;
        zipper::MatrixSpan<double, 4, 5, true> M(
            std::span<double, 20>(data.data(), data.data() + 20));
        zipper::TensorSpan_<double, et, true> T(
            std::span<double, 20>(data.data(), data.data() + 20));
        zipper::FormSpan_<double, et, true> F(
            std::span<double, 20>(data.data(), data.data() + 20));

        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 5; ++k) {
                CHECK(&data[j + k * 4] == &M(j, k));
                CHECK(&data[j + k * 4] == &T(j, k));
                CHECK(&data[j + k * 4] == &F(j, k));
            }
        }
    }
}
