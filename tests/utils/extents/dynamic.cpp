#include <zipper/detail/extents/dynamic_extents_indices.hpp>

#include "../../catch_include.hpp"

TEST_CASE("test_extent_dynamic_indices", "[extents]") {
    {
        using AE = zipper::extents<3>;
        using DEI = zipper::detail::extents::DynamicExtentIndices<AE>;
        auto indices = DEI::get_dynamic_local_indices();
        using DIT = std::decay_t<decltype(indices)>;
        static_assert(std::tuple_size_v<DIT> == 1);
        CHECK(indices[0] == zipper::dynamic_extent);
    }
    {
        using AE = zipper::extents<3, 4>;
        using DEI = zipper::detail::extents::DynamicExtentIndices<AE>;
        auto indices = DEI::get_dynamic_local_indices();
        using DIT = std::decay_t<decltype(indices)>;
        static_assert(std::tuple_size_v<DIT> == 2);
        CHECK(indices[0] == zipper::dynamic_extent);
        CHECK(indices[1] == zipper::dynamic_extent);
    }

    {
        using AE = zipper::extents<3, std::dynamic_extent>;
        std::cout << AE::static_extent(0) << " " << AE::static_extent(1) << std::endl;
        using DEI = zipper::detail::extents::DynamicExtentIndices<AE>;
        auto indices = DEI::get_dynamic_local_indices();
        using DIT = std::decay_t<decltype(indices)>;
        static_assert(std::tuple_size_v<DIT> == 2);
        CHECK(indices[0] == zipper::dynamic_extent);
        CHECK(indices[1] == 0);
    }
}
