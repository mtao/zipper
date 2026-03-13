#if !defined(ZIPPER_UTILS_EXTENTS_EXTENTS_FORMATTER_HPP)
#define ZIPPER_UTILS_EXTENTS_EXTENTS_FORMATTER_HPP
#include "zipper/detail/fmt.hpp"
#include "as_array.hpp"
#include "zipper/types.hpp"

template <zipper::index_type... Extents>
struct fmt::formatter<zipper::extents<Extents...>> : fmt::formatter<std::string> {
    auto format(const zipper::extents<Extents...>& e, fmt::format_context& ctx) const
        -> fmt::format_context::iterator {
        return fmt::formatter<std::string>::format(
            fmt::format("extents({})",
                        fmt::join(zipper::utils::extents::as_array(e), ",")),
            ctx);
    }
};

#endif
