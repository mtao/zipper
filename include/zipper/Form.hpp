#if !defined(ZIPPER_FORM_HPP)
#define ZIPPER_FORM_HPP

/// @file Form.hpp
/// @brief Owning dense row vector (1-form) type with static or dynamic extent.
/// @ingroup user_types
///
/// `Form<T, N>` represents a row vector (covector / 1-form) that contracts
/// with a `Vector` to produce a scalar.  It owns its data via `MDArray` and
/// inherits the full `FormBase` interface.
///
/// The key semantic difference between `Form` and `Vector` is their behaviour
/// under multiplication:
///   - `Form<T, N> * Vector<T, N>` produces a scalar (contraction / dot product).
///   - `Vector<T, N> * Form<T, N>` produces an outer product (rank-2 matrix).
///
/// Template parameters:
///   - `T`: scalar type (e.g. `double`, `float`).
///   - `N`: number of elements (`dynamic_extent` for runtime-sized).
///
/// @code
///   // Static 3-form
///   Form<double, 3> f({1.0, 2.0, 3.0});
///
///   // Contraction with a vector (produces a scalar)
///   Vector<double, 3> v({4.0, 5.0, 6.0});
///   double result = (f * v);  // 1*4 + 2*5 + 3*6 = 32
///
///   // Row of a matrix returns a FormBase
///   Matrix<double, 3, 3> A({...});
///   auto row0 = A.row(0);  // FormBase wrapping a slice expression
/// @endcode
///
/// @note `Form` is implemented as a type alias for `detail::Form_<T, extents<N>>`.
///       The actual class template is `detail::Form_`.
///
/// @see zipper::FormBase — CRTP base providing the 1-form interface.
/// @see zipper::Vector — owning column vector type (dual to Form).
/// @see zipper::Matrix — owning matrix type (rows are accessed as FormBase).
/// @see zipper::expression::nullary::MDArray — the underlying owning storage.

#include "FormBase.hxx"
#include "concepts/Form.hpp"
#include "detail/extents_check.hpp"
#include "zipper/expression/nullary/MDArray.hpp"
#include "zipper/expression/nullary/MDSpan.hpp"
#include "zipper/types.hpp"
namespace zipper {

namespace detail {
template <typename ValueType, concepts::Extents Extents,
          bool LeftMajor = true>
class Form_ : public FormBase<expression::nullary::MDArray<
                  ValueType, Extents, storage::tensor_layout<LeftMajor>,
                  default_accessor_policy<ValueType>>> {
public:
  using layout_type = storage::tensor_layout<LeftMajor>;
  using expression_type =
      expression::nullary::MDArray<ValueType, Extents, layout_type,
                                   default_accessor_policy<ValueType>>;
  using Base = FormBase<expression_type>;
  using Base::expression;
  using value_type = Base::value_type;
  using extents_type = Base::extents_type;
  using span_expression_type =
      expression::nullary::MDSpan<ValueType, Extents, layout_type,
                                  default_accessor_policy<ValueType>>;
  using const_span_expression_type =
      expression::nullary::MDSpan<const ValueType, Extents, layout_type,
                                  default_accessor_policy<ValueType>>;
  using span_type = FormBase<span_expression_type>;
  using const_span_type = FormBase<const_span_expression_type>;

  using Base::Base;
  using Base::extent;
  using Base::extents;

  Form_() = default;
  Form_(const Form_ &o) = default;
  Form_(Form_ &&o) = default;
  auto operator=(const Form_ &o) -> Form_ & = default;
  template <concepts::Expression Other>
  Form_(const Other &other) : Base(other) {}
  template <concepts::Expression Other>
    requires(!std::is_same_v<std::decay_t<Other>, Form_>)
  Form_(Other &&other) : Base(static_cast<const Other&>(other)) {}
  template <concepts::Form Other>
  Form_(const Other &other) : Base(other) {}
  template <concepts::Index... Args>
  Form_(Args &&...args)
    requires(sizeof...(Args) == Extents::rank() ||
             sizeof...(Args) == Extents::rank_dynamic())
      : Base(Extents(std::forward<Args>(args)...)) {
    if constexpr (sizeof...(Args) == Extents::rank()) {
      detail::check_extents<Extents>(static_cast<index_type>(args)...);
    }
  }
  template <index_type... indices>
  Form_(const zipper::extents<indices...> &e) : Base(e) {}
  auto operator=(Form_ &&o) -> Form_ & {
    expression().operator=(std::move(o.expression()));
    return *this;
  }
  using Base::operator=;
};
} // namespace detail
template <typename ValueType, index_type... Indxs>
using Form = detail::Form_<ValueType, zipper::extents<Indxs...>>;

namespace concepts::detail {

template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
struct IsForm<zipper::detail::Form_<ValueType, Extents, LeftMajor>>
    : std::true_type {};
template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
struct IsZipperBase<zipper::detail::Form_<ValueType, Extents, LeftMajor>>
    : std::true_type {};

} // namespace concepts::detail
} // namespace zipper

#endif
