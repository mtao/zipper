#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <iostream>
#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
// #include <zipper/views/nullary/ConstantView.hpp>
#include <zipper/views/nullary/IdentityView.hpp>
#include <zipper/views/nullary/RandomView.hpp>
// #include <zipper/views/nullary/UnitView.hpp>
// #include <zipper/views/reductions/Determinant.hpp>
#include <zipper/views/reductions/CoefficientSum.hpp>
#include <zipper/views/reductions/Trace.hpp>
#include <zipper/views/unary/PartialReductionView.hpp>
// #include <zipper/Vector.hpp>

namespace {
void print2(auto const& M) {
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
using namespace zipper;

namespace {
template <typename ViewType, int index>
auto make(const ViewType& a, std::integral_constant<int, index>) {
    return views::unary::PartialReductionView<
        ViewType, views::reductions::CoefficientSum, index>(a);
}
}  // namespace

TEST_CASE("test_partial_sum", "[matrix][storage][dense]") {
    {
        using IType = views::nullary::IdentityView<double, 3, 3>;
        IType i;

        auto pr = make(i, std::integral_constant<int, 1>{});

        REQUIRE(pr.extents().rank() == 1);
        REQUIRE(pr.static_extent(0) == 3);
        REQUIRE(pr.extent(0) == 3);

        CHECK(pr(0) == 1);
        CHECK(pr(1) == 1);
        CHECK(pr(2) == 1);
    }
    {
        Matrix<double, 3, 4> A = views::nullary::IdentityView<double, 3, 4>();

        auto pr = make(A.view(), std::integral_constant<int, 1>{});

        REQUIRE(pr.extents().rank() == 1);
        REQUIRE(pr.static_extent(0) == 3);
        REQUIRE(pr.extent(0) == 3);

        auto pr2 = make(A.view(), std::integral_constant<int, 0>{});

        REQUIRE(pr2.extents().rank() == 1);
        REQUIRE(pr2.static_extent(0) == 4);
        REQUIRE(pr2.extent(0) == 4);

        // slice of rows
        CHECK(pr(0) == 1);
        CHECK(pr(1) == 1);
        CHECK(pr(2) == 1);

        // slice of cols
        CHECK(pr2(0) == 1);
        CHECK(pr2(1) == 1);
        CHECK(pr2(2) == 1);
        CHECK(pr2(3) == 0);

        // update a value
        A(0, 3) = 10;
        // slice of rows
        CHECK(pr(0) == 11);
        CHECK(pr(1) == 1);
        CHECK(pr(2) == 1);

        // slice of cols
        CHECK(pr2(0) == 1);
        CHECK(pr2(1) == 1);
        CHECK(pr2(2) == 1);
        CHECK(pr2(3) == 10);

        A(2, 1) += 5;
        // slice of rows
        CHECK(pr(0) == 11);
        CHECK(pr(1) == 1);
        CHECK(pr(2) == 1 + 5);

        // slice of cols
        CHECK(pr2(0) == 1);
        CHECK(pr2(1) == 1 + 5);
        CHECK(pr2(2) == 1);
        CHECK(pr2(3) == 10);
    }
    {
        Matrix<double, 3, 3> A =
            views::nullary::uniform_random_view<double>({});
        Matrix<double, 3, 3> B =
            views::nullary::uniform_random_view<double>({});

        auto C = A * B;

        auto D = (as_tensor(A) * as_tensor(B)).eval();
        const auto& v = D.view();

        auto pr = views::unary::PartialReductionView<
            std::decay_t<decltype(v)>, views::reductions::Trace, 1, 2>(v);

        spdlog::info("Partial reduction");
        print2(pr);

        ArrayBase ar(pr);
        ArrayBase c(C.view());

        CHECK((ar == c).all());
    }
}

