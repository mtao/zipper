#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <zipper/Matrix.hpp>
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
        zipper::VectorSpan<double> v(data);

        REQUIRE(v.extent(0) == data.size());
        CHECK(&data[0] == &v(0));
        CHECK(&data[1] == &v(1));
        CHECK(&data[2] == &v(2));
    }

    {
        std::vector<double> data(20);

        zipper::VectorSpan<double> v(data);
        REQUIRE(v.extent(0) == data.size());
        for (size_t j = 0; j < data.size(); ++j) {
            CHECK(&data[j] == &v(j));
        }
    }
    {
        std::array<double, 20> data;
        zipper::MatrixSpan<double, 4, 5> v(data);
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 5; ++k) {
                CHECK(&data[j * 5 + k] == &v(j, k));
            }
        }
    }

    {
        std::vector<double> data(20);

        zipper::MatrixSpan<double, 4, 5, false> v(
            std::span<double, 20>(data.data(), data.data() + 20));
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 5; ++k) {
                CHECK(&data[j * 5 + k] == &v(j, k));
            }
        }
    }
    {
        std::vector<double> data(20);

        zipper::MatrixSpan<double, 4, 5, true> v(
            std::span<double, 20>(data.data(), data.data() + 20));
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 5; ++k) {
                CHECK(&data[j + k * 4] == &v(j, k));
            }
        }
    }
}
