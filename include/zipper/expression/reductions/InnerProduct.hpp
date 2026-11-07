#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_INNERPRODUCT_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_INNERPRODUCT_HPP

#include "ReductionBase.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"

#include <tuple>
#include <type_traits>

namespace zipper::expression::reductions {

/// Sum over all indices of expr(i...) * weights(i...): the inner product of
/// two tensors of the same shape (the Frobenius product for matrices).
/// Partial reductions use it to contract axes with a weight tensor.
template <typename Expr, typename Weights>
class InnerProduct : public ReductionBase<InnerProduct<Expr, Weights>, Expr> {
  public:
    using Base = ReductionBase<InnerProduct<Expr, Weights>, Expr>;
    using typename Base::expression_traits;
    using element_type = typename expression_traits::element_type;
    using Base::expression;

    InnerProduct(const Expr &expr, const std::remove_cvref_t<Weights> &weights)
      : Base(expr), m_weights(weights) {}

    auto operator()() const -> element_type {
        ZIPPER_ASSERT(expression().extents() == m_weights.extents());
        element_type v{0};
        for (const auto &i :
             zipper::utils::extents::all_extents_indices(expression().extents())) {
            v += std::apply(expression(), i) * std::apply(m_weights, i);
        }
        return v;
    }

  private:
    const std::remove_cvref_t<Weights> &m_weights;
};

namespace detail {
    /// Binds the weight type for use as a one-parameter reduction template.
    template <typename Weights>
    struct inner_product_binder {
        template <typename Expr>
        using reduction = InnerProduct<Expr, Weights>;
    };
} // namespace detail

} // namespace zipper::expression::reductions
#endif
