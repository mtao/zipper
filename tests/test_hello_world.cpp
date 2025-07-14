#include <spdlog/spdlog.h>

#include "catch_include.hpp"
#include <zipper/storage/PlainObjectStorage.hpp>
#include <zipper/views/binary/ArithmeticViews.hpp>
#include <zipper/views/unary/ScalarArithmeticViews.hpp>
#include <zipper/views/binary/MatrixProductView.hpp>
#include <zipper/views/unary/CastView.hpp>

TEST_CASE("test_storage", "[storage][dense]") {
    zipper::storage::PlainObjectStorage<double, zipper::extents<4, 4>> a;

    auto as = a.as_std_span();
    zipper::storage::PlainObjectStorage<
        double, zipper::extents<4, std::experimental::dynamic_extent>>
        b(zipper::extents<4, std::experimental::dynamic_extent>{4});

    spdlog::info("A size: {} {} {}", a.accessor().container().size(),
                 decltype(a)::extents_traits::static_size,
                 zipper::detail::ExtentsTraits<zipper::extents<4, 4>>::static_size);
    auto bs = b.as_std_span();

    int j = 0;
    for (auto& v : bs) {
        v = j++;
    }
    fmt::print("{}\n", fmt::join(bs, ","));

    for (zipper::index_type j = 0; j < a.extent(0); ++j) {
        for (zipper::index_type k = 0; k < a.extent(1); ++k) {
            double v = b(j, k);
            spdlog::warn("{} {} {}", j, k, v);
            a(j, k) = b(j, k);
        }
    }
    fmt::print("{}\n", fmt::join(as, ","));

    zipper::views::unary::ScalarMultipliesView<double, decltype(a), false> spv(2.0, a);
    // zipper::views::binary::AdditionView av(a,b);

    for (zipper::index_type j = 0; j < spv.extent(0); ++j) {
        for (zipper::index_type k = 0; k < spv.extent(1); ++k) {
            fmt::print("{} ", spv(j, k));
        }
        fmt::print("\n");
    }

    // TODO: deduction guides should remove templating but clang is unhappy
    zipper::views::binary::PlusView<decltype(a),decltype(b)> av(a, b);

    for (zipper::index_type j = 0; j < av.extent(0); ++j) {
        for (zipper::index_type k = 0; k < av.extent(1); ++k) {
            spdlog::warn("{} {} {}", j, k, av(j, k));
        }
    }

    // TODO: deduction guides should remove templating but clang is unhappy
    zipper::views::binary::MultipliesView<decltype(a),decltype(a)> pv(a, a);

    for (zipper::index_type j = 0; j < pv.extent(0); ++j) {
        for (zipper::index_type k = 0; k < pv.extent(1); ++k) {
            spdlog::warn("{} {} {}", j, k, pv(j, k));
        }
    }

    static_assert(zipper::concepts::MatrixViewDerived<decltype(a)>);
    zipper::views::binary::MatrixProductView mv(a, a);

    for (zipper::index_type j = 0; j < mv.extent(0); ++j) {
        for (zipper::index_type k = 0; k < mv.extent(1); ++k) {
            fmt::print("{} ", mv(j, k));
        }
        fmt::print("\n");
    }
    zipper::views::unary::CastView<int, decltype(a)> cv(a);

    for (zipper::index_type j = 0; j < cv.extent(0); ++j) {
        for (zipper::index_type k = 0; k < cv.extent(1); ++k) {
            fmt::print("{} ", cv(j, k));
        }
        fmt::print("\n");
    }

    b.assign(pv);
    spdlog::info("B extent: {}", b.extent(1));
    for (zipper::index_type j = 0; j < b.extent(0); ++j) {
        for (zipper::index_type k = 0; k < b.extent(1); ++k) {
            fmt::print("{} ", b(j, k));
        }
        fmt::print("\n");
    }
}
