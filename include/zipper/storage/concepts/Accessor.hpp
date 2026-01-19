#if !defined(ZIPPER_STORAGE_CONCEPTS_ACCESSOR_HPP)
#define ZIPPER_STORAGE_CONCEPTS_ACCESSOR_HPP
#include <concepts>
#include <cstdlib>
#include <zipper/concepts/Extents.hpp>
#include <zipper/concepts/Index.hpp>
#include <zipper/types.hpp>

namespace zipper::storage::concepts {
namespace detail {

template <index_type N, typename T>
auto call(auto &&f, T &&t) -> decltype(auto) {
  return f(t, std::make_integer_sequence<index_type, N>{});
}
template <typename T>
concept decays_to_extents = zipper::concepts::Extents<std::decay_t<T>>;
} // namespace detail

template <typename T>
concept Accessor = // requires(T &t) {} &&
    requires(const T &t, rank_type r) {
      // shape of the data
      //
      { t.extents() } -> detail::decays_to_extents;
      { t.extent(r) } -> std::same_as<index_type>;
      // size of the data
      { t.size() } -> std::same_as<std::size_t>;
    } && requires(T &t) {
      // const reference access using a set of index_types
      {
        detail::call<T::extents_type::rank()>(
            []<index_type... N>(T & t, std::integer_sequence<index_type, N...>)
                -> decltype(auto) { return t.coeff_ref(N...); },
            t)
      } -> std::same_as<typename T::value_type &>;
    } && requires(const T &t) {
      // coefficient by value using index_types
      {
        detail::call<T::extents_type::rank()>(
            []<index_type... N>(const T & t,
                                std::integer_sequence<index_type, N...>)
                -> decltype(auto) { return t.coeff(N...); },
            t)
      } -> std::same_as<typename T::value_type>;
      // const reference access using index_types
      {
        detail::call<T::extents_type::rank()>(
            []<index_type... N>(const T & t,
                                std::integer_sequence<index_type, N...>)
                -> decltype(auto) { return t.const_coeff_ref(N...); },
            t)
      } -> std::same_as<const typename T::value_type &>;
    };
} // namespace zipper::storage::concepts

#endif
