#if !defined(ZIPPER_EXPRESSION_NULLARY_ZERO_HPP)
#define ZIPPER_EXPRESSION_NULLARY_ZERO_HPP

/// @file Zero.hpp
/// @brief Nullary expression that is structurally zero everywhere.
/// @ingroup expressions_nullary
///
/// `Zero<T, Indices...>` is a rank-N expression whose every coefficient is
/// `T{}`, for any element type T (including ones that cannot be template
/// arguments, e.g. std::complex). It owns no data.
///
/// **Zero-aware sparsity:** it has `has_index_set = true` and returns empty
/// index sets, so zero-aware operations skip it and assigning it into sparse
/// storage clears the storage instead of visiting every index.
///
/// @code
///   auto z = Zero<double, 3>();                   // static 3-vector
///   auto zd = Zero<double, dynamic_extent>(n);    // dynamic
///   auto zc = Zero<std::complex<double>, 2, 2>(); // any T
/// @endcode

#include "zipper/expression/ExpressionBase.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/expression/detail/IndexSet.hpp"

namespace zipper::expression {
namespace nullary {

    template <typename T, index_type... Indices>
    class Zero
      : public ExpressionBase<Zero<T, Indices...>>
      , public zipper::extents<Indices...> {
      public:
        using self_type = Zero<T, Indices...>;
        using traits = zipper::expression::detail::ExpressionTraits<self_type>;
        using extents_type = typename traits::extents_type;
        using extents_traits = typename traits::extents_traits;
        using value_type = typename traits::value_type;

        using extents_type::extent;
        using extents_type::rank;
        auto extents() const -> const extents_type & { return *this; }

        Zero(const Zero &) = default;
        Zero(Zero &&) = default;
        auto operator=(const Zero &) -> Zero & = default;
        auto operator=(Zero &&) -> Zero & = default;

        Zero()
            requires(extents_traits::is_static)
        = default;

        Zero(const extents_type &e) : extents_type(e) {}

        template <zipper::concepts::Index... Args>
        Zero(Args &&...args)
          : Zero(extents_type(std::forward<Args>(args)...)) {}

        /// Every coefficient is value_type{}.
        auto coeff(zipper::concepts::Index auto &&...) const -> value_type {
            return value_type{};
        }

        /// Zero owns no data — make_owned() returns a copy.
        auto make_owned() const -> Zero { return *this; }

        // ── Index sets: empty along every dimension ─────────────────────

        template <rank_type D>
            requires(D < extents_type::rank())
        [[nodiscard]] auto index_set(index_type /*other_idx*/ = 0) const
            -> zipper::expression::detail::EmptyIndexRange {
            return {};
        }

        template <rank_type D>
            requires(D < extents_type::rank())
        [[nodiscard]] auto nonzero_range(index_type other_idx = 0) const
            -> zipper::expression::detail::EmptyIndexRange {
            return index_set<D>(other_idx);
        }

        [[nodiscard]] auto col_range_for_row(index_type /*row*/) const
            -> zipper::expression::detail::EmptyIndexRange
            requires(extents_type::rank() == 2)
        {
            return {};
        }

        [[nodiscard]] auto row_range_for_col(index_type /*col*/) const
            -> zipper::expression::detail::EmptyIndexRange
            requires(extents_type::rank() == 2)
        {
            return {};
        }
    };

} // namespace nullary

template <typename T, index_type... Indices>
struct detail::ExpressionTraits<nullary::Zero<T, Indices...>>
  : public BasicExpressionTraits<
        T,
        zipper::extents<Indices...>,
        expression::detail::AccessFeatures::const_value(),
        expression::detail::ShapeFeatures::resizable()> {
    /// Structurally zero everywhere.
    constexpr static bool has_index_set = true;
};

} // namespace zipper::expression

#endif
