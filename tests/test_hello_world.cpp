#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <uvl/storage/PlainObjectStorage.hpp>
#include <uvl/views/binary/AdditionView.hpp>
#include <uvl/views/binary/CoeffProductView.hpp>
#include <uvl/views/binary/MatrixProductView.hpp>
#include <uvl/views/unary/CastView.hpp>
#include <uvl/views/unary/ScalarProductView.hpp>

TEST_CASE("test_storage", "[storage][dense]") {
    uvl::storage::PlainObjectStorage<double, uvl::extents<4, 4>> a;

    auto as = a.as_span();
    uvl::storage::PlainObjectStorage<
        double, uvl::extents<4, std::experimental::dynamic_extent>>
        b(uvl::extents<4, std::experimental::dynamic_extent>{4});

    spdlog::info("A size: {} {} {}", a.accessor().container().size(),
                 decltype(a)::extents_traits::static_size,
                 uvl::detail::ExtentsTraits<uvl::extents<4, 4>>::static_size);
    auto bs = b.as_span();

    int j = 0;
    for (auto& v : bs) {
        v = j++;
    }
    fmt::print("{}\n", fmt::join(bs, ","));

    for (uvl::index_type j = 0; j < a.extent(0); ++j) {
        for (uvl::index_type k = 0; k < a.extent(1); ++k) {
            double v = b(j, k);
            spdlog::warn("{} {} {}", j, k, v);
            a(j, k) = b(j, k);
        }
    }
    fmt::print("{}\n", fmt::join(as, ","));

    uvl::views::unary::ScalarProductView spv(2.0, a);
    // uvl::views::binary::AdditionView av(a,b);

    for (uvl::index_type j = 0; j < spv.extent(0); ++j) {
        for (uvl::index_type k = 0; k < spv.extent(1); ++k) {
            fmt::print("{} ", spv(j, k));
        }
        fmt::print("\n");
    }

    uvl::views::binary::AdditionView av(a, b);

    for (uvl::index_type j = 0; j < av.extent(0); ++j) {
        for (uvl::index_type k = 0; k < av.extent(1); ++k) {
            spdlog::warn("{} {} {}", j, k, av(j, k));
        }
    }
    uvl::views::binary::CoeffProductView pv(a, a);

    for (uvl::index_type j = 0; j < pv.extent(0); ++j) {
        for (uvl::index_type k = 0; k < pv.extent(1); ++k) {
            spdlog::warn("{} {} {}", j, k, pv(j, k));
        }
    }

    static_assert(uvl::concepts::MatrixViewDerived<decltype(a)>);
    uvl::views::binary::MatrixProductView mv(a, a);

    for (uvl::index_type j = 0; j < mv.extent(0); ++j) {
        for (uvl::index_type k = 0; k < mv.extent(1); ++k) {
            fmt::print("{} ", mv(j, k));
        }
        fmt::print("\n");
    }
    uvl::views::unary::CastView<int, decltype(a)> cv(a);

    for (uvl::index_type j = 0; j < cv.extent(0); ++j) {
        for (uvl::index_type k = 0; k < cv.extent(1); ++k) {
            fmt::print("{} ", cv(j, k));
        }
        fmt::print("\n");
    }

    b.assign(pv);
    spdlog::info("B extent: {}", b.extent(1));
    for (uvl::index_type j = 0; j < b.extent(0); ++j) {
        for (uvl::index_type k = 0; k < b.extent(1); ++k) {
            fmt::print("{} ", b(j, k));
        }
        fmt::print("\n");
    }
}
