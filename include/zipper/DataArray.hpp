#if !defined(ZIPPER_DATAARRAY_HPP)
#define ZIPPER_DATAARRAY_HPP

#include "DataArrayBase.hxx"
#include "concepts/DataArray.hpp"
#include "concepts/Extents.hpp"
#include "detail/assert.hpp"
#include "detail/extents_check.hpp"
#include "zipper/expression/nullary/MDArray.hpp"
#include "zipper/expression/nullary/MDSpan.hpp"
#include "zipper/types.hpp"

namespace zipper {

namespace detail {

template <typename ValueType, concepts::Extents Extents,
          bool LeftMajor = true>
class DataArray_
    : public DataArrayBase<expression::nullary::MDArray<
          ValueType, Extents, storage::tensor_layout<LeftMajor>,
          default_accessor_policy<ValueType>>> {
public:
  using layout_type = storage::tensor_layout<LeftMajor>;
  using expression_type =
      expression::nullary::MDArray<ValueType, Extents, layout_type,
                                   default_accessor_policy<ValueType>>;
  using Base = DataArrayBase<expression_type>;
  using Base::expression;
  using value_type = Base::value_type;
  using extents_type = Base::extents_type;
  using Base::extent;
  using Base::extents;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  constexpr static bool is_static = extents_traits::is_static;

  using span_expression_type =
      expression::nullary::MDSpan<ValueType, Extents, layout_type,
                                  default_accessor_policy<ValueType>>;
  using const_span_expression_type =
      expression::nullary::MDSpan<const ValueType, Extents, layout_type,
                                  default_accessor_policy<ValueType>>;
  using span_type = DataArrayBase<span_expression_type>;
  using const_span_type = DataArrayBase<const_span_expression_type>;

  DataArray_() = default;
  DataArray_(const DataArray_ &o) = default;
  DataArray_(DataArray_ &&o) = default;
  auto operator=(const DataArray_ &o) -> DataArray_ & = default;
  auto operator=(DataArray_ &&o) -> DataArray_ & {
    expression().operator=(std::move(o.expression()));
    return *this;
  }

  template <concepts::Expression Other>
  DataArray_(const Other &other) : Base(other) {}

  template <concepts::DataArray Other>
  DataArray_(const Other &other) : Base(other) {}

  template <typename... Args>
  DataArray_(Args &&...args)
    requires((std::is_convertible_v<Args, index_type> && ...) &&
             (sizeof...(Args) == Extents::rank() ||
              sizeof...(Args) == Extents::rank_dynamic()))
      : Base(Extents(std::forward<Args>(args)...)) {
    if constexpr (sizeof...(Args) == Extents::rank()) {
      detail::check_extents<Extents>(static_cast<index_type>(args)...);
    }
  }

  template <index_type... indices>
  DataArray_(const zipper::extents<indices...> &e) : Base(e) {}

  auto as_span() -> span_type {
    if constexpr (is_static) {
      return span_type(expression().as_std_span());
    } else {
      return span_type(expression().as_std_span(), extents());
    }
  }
  auto as_const_span() const -> const_span_type {
    if constexpr (is_static) {
      return const_span_type(expression().as_std_span());
    } else {
      return const_span_type(expression().as_std_span(), extents());
    }
  }
  auto as_span() const -> const_span_type { return as_const_span(); }

  auto begin() { return expression().linear_accessor().begin(); }
  auto end() { return expression().linear_accessor().end(); }
  auto begin() const { return expression().linear_accessor().begin(); }
  auto end() const { return expression().linear_accessor().end(); }

  auto data() { return expression().data(); }
  auto data() const { return expression().data(); }

  using Base::operator=;

  /// Returns a zero-initialized DataArray (static extents only).
  static auto zero() -> DataArray_
    requires(is_static)
  {
    DataArray_ result;
    result.fill(value_type{});
    return result;
  }

  /// Returns a zero-initialized DataArray (dynamic extents).
  /// Pass the dynamic extent sizes as arguments.
  template <typename... Args>
  static auto zero(Args &&...args) -> DataArray_
    requires(!is_static && (std::is_convertible_v<Args, index_type> && ...))
  {
    DataArray_ result(std::forward<Args>(args)...);
    result.fill(value_type{});
    return result;
  }

  /// Returns a new DataArray with reinterpreted extents.
  /// The total number of elements must match.  Static extents are
  /// checked at compile time; dynamic extents are checked at runtime
  /// via ZIPPER_ASSERT.
  template <concepts::Extents NewExtents>
  auto reshape(const NewExtents &new_ext = NewExtents{}) const
      -> DataArray_<value_type, NewExtents, LeftMajor> {
    using new_traits = zipper::detail::ExtentsTraits<NewExtents>;
    if constexpr (is_static && new_traits::is_static) {
      static_assert(extents_traits::static_size == new_traits::static_size,
                    "reshape: total element count must match");
    } else {
      ZIPPER_ASSERT(new_traits::size(new_ext) ==
                    extents_traits::size(extents()));
    }
    DataArray_<value_type, NewExtents, LeftMajor> result(new_ext);
    std::copy(begin(), end(), result.begin());
    return result;
  }
};

template <concepts::DataArray CB>
DataArray_(const CB &o)
    -> DataArray_<typename CB::value_type, typename CB::extents_type>;

} // namespace detail

template <typename ValueType, index_type... Indxs>
using DataArray = detail::DataArray_<ValueType, zipper::extents<Indxs...>>;

namespace concepts::detail {

template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
struct IsDataArray<zipper::detail::DataArray_<ValueType, Extents, LeftMajor>>
    : std::true_type {};
template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
struct IsZipperBase<zipper::detail::DataArray_<ValueType, Extents, LeftMajor>>
    : std::true_type {};

} // namespace concepts::detail

} // namespace zipper
#endif
