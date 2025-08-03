
#include <iostream>
#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>

#include "../../catch_include.hpp"
#include "../../fmt_include.hpp"
// #include <zipper/views/nullary/ConstantView.hpp>
#include <zipper/views/nullary/IdentityView.hpp>
#include <zipper/views/unary/CastView.hpp>
#include <zipper/utils/format.hpp>
#include <zipper/views/reductions/CoefficientSum.hpp>
#include <zipper/views/reductions/Trace.hpp>
#include <zipper/views/unary/PartialReductionView.hpp>
#include <zipper/views/unary/detail/PartialReductionViewDispatcher.hpp>

// #include <zipper/Vector.hpp>

using namespace zipper;


TEST_CASE("test_partial_sum", "[views][unary]") {
    {
        using IType = views::nullary::IdentityView<double, 3, 3>;
        IType i;

        auto i2 = views::unary::cast<float>(i);
        CHECK(i2(0,0) == 1.0f);
        CHECK(i2(1,0) == 0.0f);
        CHECK(i2(2,0) == 0.0f);
        CHECK(i2(0,1) == 0.0f);
        CHECK(i2(1,1) == 1.0f);
        CHECK(i2(2,1) == 0.0f);
        CHECK(i2(0,2) == 0.0f);
        CHECK(i2(1,2) == 0.0f);
        CHECK(i2(2,2) == 1.0f);

    }
    {
        Matrix<double, 3, 3> A = views::nullary::IdentityView<double, 3, 3>();

        auto i2 = A.cast<float>();
        CHECK(i2(0,0) == 1.0f);
        CHECK(i2(1,0) == 0.0f);
        CHECK(i2(2,0) == 0.0f);
        CHECK(i2(0,1) == 0.0f);
        CHECK(i2(1,1) == 1.0f);
        CHECK(i2(2,1) == 0.0f);
        CHECK(i2(0,2) == 0.0f);
        CHECK(i2(1,2) == 0.0f);
        CHECK(i2(2,2) == 1.0f);

    }
}

