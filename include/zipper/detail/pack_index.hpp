#if !defined(ZIPPER_DETAIL_PACK_INDEX_HPP)
#define ZIPPER_DETAIL_PACK_INDEX_HPP

#include <utility>
namespace zipper::detail {
template <std::size_t K, typename Arg0, typename... Args>
constexpr auto pack_index(Arg0&& arg, Args&&... args) -> decltype(auto)
    requires(K <= sizeof...(Args))

{
    if constexpr (K == 0) {
        return std::forward<Arg0>(arg);
    } else {
        return pack_index<K - 1>(std::forward<Args>(args)...);
    }
}
}  // namespace zipper::detail
#endif
