#if !defined(ZIPPER_STORAGE_CONCEPTS_DATA_HPP)
#define ZIPPER_STORAGE_CONCEPTS_DATA_HPP
#include <concepts>
#include <cstdlib>
#include <zipper/types.hpp>

namespace zipper::storage::concepts {

// NOTE: The below use of convertible_to is an idiosyncrasy on gcc because
// StdSpan is extends a std::span, which returns a __gnu_cxx::__normal_iterator
// rather than a std::basic_iterator<> sort of object
template <typename T>
concept Data = std::same_as<std::remove_cv_t<typename T::element_type>,
                            typename T::value_type> &&
               requires(T t) {
                 // mutable iterators
                 {
                   t.begin()
                 } -> std::convertible_to<typename T::iterator_type>;
                 { t.end() } -> std::convertible_to<typename T::iterator_type>;
               } && requires(const T t) {
                 // size of the data
                 { t.size() } -> std::convertible_to<std::size_t>;

                 // const iterator types
                 {
                   t.begin()
                 } -> std::convertible_to<typename T::const_iterator_type>;
                 {
                   t.end()
                 } -> std::convertible_to<typename T::const_iterator_type>;
                 {
                   t.cbegin()
                 } -> std::convertible_to<typename T::const_iterator_type>;
                 {
                   t.cend()
                 } -> std::convertible_to<typename T::const_iterator_type>;
               } && (T::is_const || (requires(T t, index_type index) {
                       // const reference access
                       {
                         t.coeff_ref(index)
                       } -> std::same_as<typename T::element_type &>;
                     })) && requires(const T t, index_type index) {
                 // coefficient by element
                 // Ideally this returns an element_type, but cppreference
                 // section on decltype on expressions states: "The type is
                 // non-const even if the entity is a template parameter object
                 // (which is a const object)." So the const qualifier is
                 // inherently implicitly lost on a
                 // requires-expression-type-constraint
                 { t.coeff(index) } -> std::same_as<typename T::value_type>;
                 // const reference access
                 {
                   t.const_coeff_ref(index)
                 } -> std::same_as<typename T::value_type const &>;
               };
} // namespace zipper::storage::concepts

#endif
