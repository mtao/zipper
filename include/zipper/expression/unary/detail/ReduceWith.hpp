#if !defined(ZIPPER_EXPRESSION_UNARY_DETAIL_REDUCEWITH_HPP)
#define ZIPPER_EXPRESSION_UNARY_DETAIL_REDUCEWITH_HPP

#include <tuple>
#include <type_traits>

namespace zipper::expression::unary::detail {

/// A PartialMap function applying the full reduction Reduction (e.g.
/// reductions::CoefficientSum) to each sub-tensor, with bound `parameters`
/// passed to its constructor after the sub-tensor.
template <template <typename> typename Reduction, typename... Parameters>
struct ReduceWith {
    std::tuple<Parameters...> parameters;

    template <typename SubTensor>
    auto operator()(const SubTensor &sub_tensor) const {
        return std::apply(
            [&](const auto &...p) { return Reduction<const SubTensor &>(sub_tensor, p...)(); },
            parameters);
    }
};

} // namespace zipper::expression::unary::detail
#endif
