#if !defined(ZIPPER_STORAGE_SPANDATA_HPP)
#define ZIPPER_STORAGE_SPANDATA_HPP
#include "LinearAccessorTraits.hpp"

#include <span>
#include <zipper/types.hpp>
namespace zipper::storage {
template <typename ElementType, std::size_t N>
class SpanData : public std::span<ElementType, N> {
public:
  using element_type = ElementType;
  using value_type = std::remove_cv_t<ElementType>;
  constexpr static bool is_const = std::is_const_v<element_type>;
  using std_span_type = std::span<element_type, N>;
  constexpr static index_type static_size = N;

  using base_type = std_span_type;
  using base_type::base_type;
  // for some reason the above using statement didn't work for copy?
  SpanData(const base_type &b) : base_type(b) {}

  /*
  template <typename ET, std::size_t M>
  SpanData(const std::span<ET, M> &o)
    requires(M != N && M == std::dynamic_extent)
      : std_span_type(o) {}
  */
  template <typename T> SpanData(const T &) {}
  using base_type::size;

  auto coeff(index_type i) const -> element_type {
    return base_type::operator[](i);
  }
  auto coeff_ref(index_type i) -> value_type &
    requires(!is_const)
  {
    return base_type::operator[](i);
  }
  auto const_coeff_ref(index_type i) const -> value_type const & {
    return base_type::operator[](i);
  }

  using base_type::data;

  auto container() const -> const base_type & { return *this; }
  auto container() -> base_type & { return *this; }

  auto as_std_span() -> base_type & { return *this; }
  auto as_std_span() const -> base_type const & { return *this; }

  using base_type::begin;
  using base_type::cbegin;
  using base_type::cend;
  using base_type::end;
  // TODO: for cpp<23 iterator_type and const_iterator_type are different.
  // feeling very lazy as this discrepency disappears with cpp23
#if defined(__cpp_lib_ranges_as_const)
  using iterator_type = base_type::iterator;
  using const_iterator_type = base_type::const_iterator;
#else
  using iterator_type = base_type::iterator;
  using const_iterator_type = base_type::iterator;
#endif
};

template <typename ElementType, std::size_t N>
struct LinearAccessorTraits<SpanData<ElementType, N>>
    : public BasicLinearAccessorTraits<
          AccessFeatures{.is_const = std::is_const_v<ElementType>,
                         .is_reference = true,
                         .is_alias_free = true},
          ShapeFeatures{.is_resizable = false}> {};
} // namespace zipper::storage
#endif
