#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <uvl/types.hpp>
#include <uvl/detail/dynamic_extent_indices.hpp>

TEST_CASE("test_extents_dynamic_size", "[extents][dense]") {

    {
    uvl::extents<5,std::dynamic_extent> E;
    constexpr static auto dyn = uvl::detail::dynamic_extent_indices_t<decltype(E)>;
    static_assert(dyn.size() == 1);
    static_assert(dyn[0] == 1);
    }
    {
    uvl::extents<std::dynamic_extent,5,std::dynamic_extent> E;
    constexpr static auto dyn = uvl::detail::dynamic_extent_indices_t<decltype(E)>;
    static_assert(dyn.size() == 2);
    static_assert(dyn[0] == 0);
    static_assert(dyn[1] == 2);
    }
}
