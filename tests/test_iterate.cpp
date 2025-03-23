
#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <uvl/detail/extents/all_extents_indices.hpp>

TEST_CASE("test_all_extents", "[storage][dense]") {
    for (const auto& ind : uvl::detail::extents::all_extents_indices(3, 4)) {
        // auto v = ind | ranges::views::transform([](auto i) { return 2 * i;
        // });

        spdlog::info("{} {}", std::get<0>(ind), std::get<1>(ind));
    }
}
