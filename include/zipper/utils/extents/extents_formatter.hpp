#if !defined(ZIPPER_UTILS_EXTENTS_EXTENTS_FORMATTER_HPP)
#define ZIPPER_UTILS_EXTENTS_EXTENTS_FORMATTER_HPP
#include <format>
#include <string>
#include "as_array.hpp"
#include "zipper/types.hpp"

template <zipper::index_type... Extents>
struct std::formatter<zipper::extents<Extents...>> : std::formatter<std::string> {
    auto format(const zipper::extents<Extents...>& e, std::format_context& ctx) const
        -> std::format_context::iterator {
        auto arr = zipper::utils::extents::as_array(e);
        std::string joined;
        for (std::size_t i = 0; i < arr.size(); ++i) {
            if (i > 0) joined += ',';
            joined += std::format("{}", arr[i]);
        }
        return std::formatter<std::string>::format(
            std::format("extents({})", joined), ctx);
    }
};

#endif
