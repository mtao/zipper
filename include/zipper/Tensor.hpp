#if !defined(ZIPPER_TENSOR_HPP)
#define ZIPPER_TENSOR_HPP

#include "TensorBase.hxx"
#include "concepts/Extents.hpp"
#include "concepts/Tensor.hpp"
#include "zipper/expression/nullary/MDArray.hpp"
#include "zipper/expression/nullary/MDSpan.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
class Tensor_ : public TensorBase<expression::nullary::MDArray<
                    ValueType, Extents, storage::tensor_layout<LeftMajor>,
                    default_accessor_policy<ValueType>>> {
   public:
    using layout_type = storage::tensor_layout<LeftMajor>;
    using expression_type =
        expression::nullary::MDArray<ValueType, Extents, layout_type,
                                     default_accessor_policy<ValueType>>;
    using Base = TensorBase<expression_type>;
    using Base::expression;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using Base::extent;
    using Base::extents;
    using span_expression_type =
        expression::nullary::MDSpan<ValueType, Extents, layout_type,
                                    default_accessor_policy<ValueType>>;
    using const_span_expression_type =
        expression::nullary::MDSpan<const ValueType, Extents, layout_type,
                                    default_accessor_policy<ValueType>>;
    using span_type = TensorBase<span_expression_type>;
    using const_span_type = TensorBase<const_span_expression_type>;

    template <concepts::Expression Other>
    Tensor_(const Other& other) : Base(other) {}
    template <concepts::Tensor Other>
    Tensor_(const Other& other) : Base(other) {}
    template <concepts::Index... Args>
    Tensor_(Args&&... args)
        : Base(Extents(std::forward<Args>(args)...)) {}
    Tensor_() = default;
    Tensor_(const Tensor_&) = default;
    Tensor_(Tensor_&&) = default;
    template <index_type... indices>
    Tensor_(const zipper::extents<indices...>& e) : Base(e) {}
    Tensor_& operator=(Tensor_&& o) {
        expression().operator=(std::move(o.expression()));
        return *this;
    }
    using Base::operator=;
};
template <typename ValueType, index_type... Indxs>
using Tensor = Tensor_<ValueType, zipper::extents<Indxs...>>;
}  // namespace zipper

#endif
