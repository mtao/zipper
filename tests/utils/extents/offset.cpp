

#include <span>
#include <zipper/concepts/shapes.hpp>
#include <zipper/detail/extents/dynamic_extents_indices.hpp>
#include <zipper/detail/extents/swizzle_extents.hpp>
#include <zipper/types.hpp>
#include <zipper/utils/extents/is_compatible.hpp>
#include <zipper/utils/extents/offset_extents.hpp>

#include "../../catch_include.hpp"

TEST_CASE("test_extent_compatibility", "[extents]") {
    {
        using AE = zipper::extents<3>;
        // using OE = zipper::extents<1>;
        using OE = std::integer_sequence<int64_t, 1>;
        using T2 = zipper::utils::extents::OffsetExtents<AE, OE>::extents_type;
        CHECK(zipper::extents<4>{} == T2{});
    }

    zipper::extents<3, 4> A;
    zipper::extents<3, std::dynamic_extent> B(3);
    {
        using AE = zipper::extents<3, 4>;
        using OE = std::integer_sequence<int64_t, 0, 1>;
        using T2 = zipper::utils::extents::OffsetExtents<AE, OE>::extents_type;
        using T =
            zipper::utils::extents::offset_extents_t<zipper::extents<3, 4>, 0,
                                                     1>;
        static_assert(std::is_same_v<zipper::extents<3, 5>, T>);
        static_assert(std::is_same_v<zipper::extents<3, 5>, T2>);

        auto r = zipper::utils::extents::offset_extents<0, 1>(A);
        static_assert(std::is_same_v<std::decay_t<decltype(r)>, T>);
    }
    {
        using T =
            zipper::utils::extents::offset_extents_t<zipper::extents<3, 4>, -1,
                                                     0>;
        static_assert(std::is_same_v<zipper::extents<2, 4>, T>);

        auto r = zipper::utils::extents::offset_extents<-1, 0>(A);
        static_assert(std::is_same_v<std::decay_t<decltype(r)>, T>);
    }
    {
        using AE = zipper::extents<3, std::dynamic_extent>;
        using OE = std::integer_sequence<int64_t, 0, 1>;
        using T2 = zipper::utils::extents::OffsetExtents<AE, OE>::extents_type;
        using T = zipper::utils::extents::offset_extents_t<
            zipper::extents<3, std::dynamic_extent>, 0, 1>;
        static_assert(
            std::is_same_v<zipper::extents<3, std::dynamic_extent>, T>);
        static_assert(
            std::is_same_v<zipper::extents<3, std::dynamic_extent>, T2>);

        auto r = zipper::utils::extents::offset_extents<0, 1>(B);
        static_assert(std::is_same_v<std::decay_t<decltype(r)>, T>);
    }
    {
        using T = zipper::utils::extents::offset_extents_t<
            zipper::extents<3, std::dynamic_extent>, -1, -1>;
        static_assert(
            std::is_same_v<zipper::extents<2, std::dynamic_extent>, T>);

        auto r = zipper::utils::extents::offset_extents<-1, -1>(B);
        static_assert(std::is_same_v<std::decay_t<decltype(r)>, T>);
        CHECK(B == zipper::create_dextents(3, 3));
        CHECK(r == zipper::create_dextents(2, 2));
    }
}
