#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <uvl/storage/PlainObjectStorage.hpp>
#include <uvl/views/binary/AdditionView.hpp>
#include <uvl/views/unary/ScalarProductView.hpp>

TEST_CASE("test_storage", "[storage][dense]") {
    uvl::storage::PlainObjectStorage<double, uvl::extents<4, 4>> a;

    auto amd = a.as_mdspan();
    auto as = a.as_span();
    uvl::storage::PlainObjectStorage<
        double, uvl::extents<4, std::experimental::dynamic_extent>>
        b(uvl::extents<4, std::experimental::dynamic_extent>{4});
    auto bmd = b.as_mdspan();
    auto bs = b.as_span();

    int j = 0;
    spdlog::info("BS SIZE: {} ", bs.size());
    for (auto& v : bs) {
        v = j++;
    }
    fmt::print("{}\n", fmt::join(bs, ","));

    for (uvl::index_type j = 0; j < a.extent(0); ++j) {
        for (uvl::index_type k = 0; k < a.extent(1); ++k) {
            std::array<uvl::index_type, 2> i{{j, k}};
            spdlog::warn("{} {} {}", j, k, bmd[i]);
            amd[i] = 2 * bmd[i];
        }
    }
    fmt::print("{}\n", fmt::join(as, ","));

    uvl::views::unary::ScalarProductView spv(2.0, a);
    // uvl::views::binary::AdditionView av(a,b);

    for (uvl::index_type j = 0; j < spv.extent(0); ++j) {
        for (uvl::index_type k = 0; k < spv.extent(1); ++k) {
            std::array<uvl::index_type, 2> i{{j, k}};
            spdlog::warn("{} {} {}", j, k, spv(i));
        }
    }
}
