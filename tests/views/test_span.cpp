#include "../fmt_include.hpp"
#include "../catch_include.hpp"
#include <zipper/Form.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/views/nullary/ConstantView.hpp>
#include <zipper/views/nullary/IdentityView.hpp>
#include <zipper/views/nullary/RandomView.hpp>
#include <zipper/views/unary/PartialTraceView.hpp>
#include <zipper/views/unary/SwizzleView.hpp>

TEST_CASE("test_span_construction", "[vector][storage][dense]") {
    {
        std::array<double, 3> data;
        zipper::Vector<double, 3>::span_type v(data);

        CHECK(&data[0] == &v(0));
        CHECK(&data[1] == &v(1));
        CHECK(&data[2] == &v(2));
    }
    {
        std::array<double, 3> data;
        zipper::Vector<double, 3>::span_type v(data);
        zipper::Tensor<double, 3>::span_type T(data);
        zipper::Form<double, 3>::span_type F(data);
        zipper::Array<double, 3>::span_type A(data);

        REQUIRE(v.extent(0) == data.size());
        static_assert(std::decay_t<decltype(v.extents())>::static_extent(0) ==
                      data.size());
        CHECK(&data[0] == &v(0));
        CHECK(&data[1] == &v(1));
        CHECK(&data[2] == &v(2));

        CHECK(&data[0] == &F(0));
        CHECK(&data[1] == &F(1));
        CHECK(&data[2] == &F(2));

        CHECK(&data[0] == &T(0));
        CHECK(&data[1] == &T(1));
        CHECK(&data[2] == &T(2));

        CHECK(&data[0] == &A(0));
        CHECK(&data[1] == &A(1));
        CHECK(&data[2] == &A(2));
    }

    {
        std::vector<double> data(20);

        zipper::VectorBase v((std::span<double>(data)));
        zipper::TensorBase T((std::span<double>(data)));
        zipper::FormBase F((std::span<double>(data)));
        zipper::ArrayBase A((std::span<double>(data)));
        REQUIRE(v.extent(0) == data.size());
        for (size_t j = 0; j < data.size(); ++j) {
            CHECK(&data[j] == &v(j));
            CHECK(&data[j] == &T(j));
            CHECK(&data[j] == &F(j));
        }
    }
    {
        std::array<double, 20> data;
        zipper::Matrix<double, 4, 5>::span_type M(data);
        zipper::Tensor<double, 4, 5>::span_type T(data);
        zipper::Form<double, 4, 5>::span_type F(data);
        zipper::Array<double, 4, 5>::span_type A(data);
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 5; ++k) {
                CHECK(&data[j * 5 + k] == &M(j, k));
                CHECK(&data[j * 5 + k] == &T(j, k));
                CHECK(&data[j * 5 + k] == &F(j, k));
                CHECK(&data[j * 5 + k] == &A(j, k));
            }
        }
    }

    {
        std::vector<double> data(20);

        using et = zipper::extents<4, 5>;
        zipper::Matrix<double, 4, 5, false>::span_type M(
            std::span<double, 20>(data.data(), data.data() + 20));
        zipper::Tensor_<double, et, false>::span_type T(
            std::span<double, 20>(data.data(), data.data() + 20));
        zipper::Form_<double, et, false>::span_type F(
            std::span<double, 20>(data.data(), data.data() + 20));
        zipper::Array_<double, et, false>::span_type A(
            std::span<double, 20>(data.data(), data.data() + 20));
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 5; ++k) {
                CHECK(&data[j + k * 4] == &M(j, k));
                CHECK(&data[j + k * 4] == &T(j, k));
                CHECK(&data[j + k * 4] == &F(j, k));
                CHECK(&data[j + k * 4] == &A(j, k));
            }
        }
    }
    {
        std::vector<double> data(20);
        std::iota(data.begin(), data.end(), 0);

        using et = zipper::extents<4, 5>;
        zipper::Matrix<double, 4, 5, true>::span_type M(
            std::span<double, 20>(data.data(), data.data() + 20));
        zipper::Tensor_<double, et, true>::span_type T(
            std::span<double, 20>(data.data(), data.data() + 20));
        zipper::Form_<double, et, true>::span_type F(
            std::span<double, 20>(data.data(), data.data() + 20));
        zipper::Array_<double, et, true>::span_type A(
            std::span<double, 20>(data.data(), data.data() + 20));
        auto fv = F.view();
        auto tv = T.view();
        auto mv = M.view();
        auto av = A.view();

        auto fs = fv.as_std_span();
        auto ts = tv.as_std_span();
        auto ms = mv.as_std_span();
        auto as = av.as_std_span();

        CHECK(fs.data() == ts.data());
        CHECK(fs.data() == ms.data());
        CHECK(as.data() == ms.data());
        return;

        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 5; ++k) {
                CHECK(&data[j * 5 + k] == &M(j, k));
                CHECK(&data[j * 5 + k] == &T(j, k));
                CHECK(&data[j * 5 + k] == &F(j, k));
                CHECK(&data[j * 5 + k] == &A(j, k));
            }
        }
    }
}
