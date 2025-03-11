#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <uvl/storage/StaticPlainObjectStorage.hpp>
#include <uvl/storage/DynamicPlainObjectStorage.hpp>

TEST_CASE("test_storage", "[storage][dense]") {
    uvl::storage::StaticPlainObjectStorage<double, uvl::extents<4, 4>> a;
    
    auto amd = a.as_mdspan();
    auto as = a.as_span();
    uvl::storage::DynamicPlainObjectStorage<
        double, uvl::extents<4, std::experimental::dynamic_extent>>
        b(uvl::extents<4, std::experimental::dynamic_extent>{4});
    auto bmd = b.as_mdspan();
    auto bs = b.as_span();

    int j = 0;
    for (auto& v : bs) {
        v = j++;
    }
    fmt::print("{}\n", fmt::join(bs, ","));

    for (uvl::index_type j = 0; j < a.extent(0); ++j) {
        for (uvl::index_type k = 0; k < a.extent(1); ++k) {
            std::array<uvl::index_type, 2> i{{j, k}};
            spdlog::warn("{} {} {}", j, k, bmd[i]);
            amd[i] = bmd[i];
        }
    }
    fmt::print("{}\n", fmt::join(as, ","));
}
