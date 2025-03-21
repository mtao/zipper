#if !defined(UVL_DETAIL_PACK_INDEX_HPP)
#define UVL_DETAIL_PACK_INDEX_HPP

#include <utility>
namespace uvl::detail {
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
}  // namespace uvl::detail
#endif
