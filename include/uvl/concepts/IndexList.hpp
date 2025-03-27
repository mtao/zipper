#if !defined(UVL_CONCEPTS_INDEXLIST_HPP)
#define UVL_CONCEPTS_INDEXLIST_HPP
#include "IndexLike.hpp"
#include "uvl/types.hpp"
#include <vector>
#include <array>
#include <map>

namespace uvl::concepts {


// A value ois slice-like if it can belong in a slice.
// That is, it's an index, or a component in a tuple
template <typename T>
concept IndexList = std::is_same_v<typename T::value_type,index_type> && requires(const T& v, std::size_t j) {v.at(j);};
}  // namespace uvl::concepts
#endif
