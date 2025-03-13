#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <uvl/types.hpp>
#include <uvl/detail/dynamic_extents_indices.hpp>

TEST_CASE("test_extents_dynamic_size", "[extents][dense]") {

    {
    uvl::extents<5,std::dynamic_extent> E(19);
    constexpr static auto dyn = uvl::detail::dynamic_extents_indices_v<decltype(E)>;
    static_assert(dyn.size() == 1);
    static_assert(dyn[0] == 1);
    auto val = uvl::detail::dynamic_extents(E);
    static_assert(val.size() == 1);
    CHECK(val[0] == 19);
    }
    {
    uvl::extents<std::dynamic_extent,5,std::dynamic_extent> E(3,5);
    constexpr static auto dyn = uvl::detail::dynamic_extents_indices_v<decltype(E)>;
    static_assert(dyn.size() == 2);
    static_assert(dyn[0] == 0);
    static_assert(dyn[1] == 2);
    auto val = uvl::detail::dynamic_extents(E);
    static_assert(val.size() == 2);
    CHECK(val[0] == 3);
    CHECK(val[1] == 5);
    }
}
