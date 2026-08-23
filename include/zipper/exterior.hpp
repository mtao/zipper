#if !defined(ZIPPER_EXTERIOR_HPP)
#define ZIPPER_EXTERIOR_HPP

#include <algorithm>
#include <array>
#include <vector>

#include "Form.hpp"
#include "Vector.hpp"
#include "detail/declare_operations.hpp"
#include "expression/binary/ExteriorProduct.hpp"
#include "expression/unary/AlternatingProjection.hpp"
#include "exterior/detail/combinatorics.hpp"

namespace zipper {

namespace exterior::detail {

template <typename ValueType, index_type AmbientDimension, rank_type Degree,
          typename Seq>
struct static_form_type;

template <typename ValueType, index_type AmbientDimension, rank_type Degree,
          std::size_t... N>
struct static_form_type<ValueType, AmbientDimension, Degree,
                        std::index_sequence<N...>> {
    using type = Form<ValueType, ((void)N, AmbientDimension)...>;
};

template <typename ValueType, index_type AmbientDimension, rank_type Degree>
using static_form_t =
    typename static_form_type<ValueType, AmbientDimension, Degree,
                              std::make_index_sequence<Degree>>::type;

template <typename Extents, typename Seq>
struct static_extents_equal;

template <typename Extents, std::size_t... N>
struct static_extents_equal<Extents, std::index_sequence<N...>>
    : std::bool_constant<((Extents::static_extent(N + 1)
                           == Extents::static_extent(0))
                          && ...)> {};

template <typename Expr, std::size_t... N>
auto evaluate_form(const Expr &expr, const std::vector<index_type> &indices,
                   std::index_sequence<N...>) {
    return expr(indices[N]...);
}

template <concepts::Form F, std::size_t... N>
auto assign_dense_coeff(F &form, const std::vector<index_type> &indices,
                        typename F::value_type value,
                        std::index_sequence<N...>) -> void {
    form(indices[N]...) = value;
}

inline auto hodge_sign(index_type ambient_dimension,
                       const std::vector<index_type> &sorted_indices) -> int {
    const auto complement =
        zipper::exterior::detail::complement_indices(ambient_dimension,
                                                     sorted_indices);
    std::vector<index_type> permutation;
    permutation.reserve(ambient_dimension);
    permutation.insert(permutation.end(), sorted_indices.begin(),
                       sorted_indices.end());
    permutation.insert(permutation.end(), complement.begin(), complement.end());
    return zipper::exterior::detail::inversion_parity(permutation);
}

} // namespace exterior::detail

template <concepts::Form Expr>
auto alternating_part(Expr &&expr) {
    using child_t = detail::forwarded_expression_t<Expr>;
    using V = expression::unary::AlternatingProjection<child_t>;
    return FormBase<V>(std::in_place, std::forward<Expr>(expr).expression());
}

template <typename Expr1, typename Expr2>
    requires(concepts::Form<std::decay_t<Expr1>> &&
             concepts::Form<std::decay_t<Expr2>>)
auto wedge(Expr1 &&lhs, Expr2 &&rhs) {
    using A = detail::forwarded_expression_t<Expr1>;
    using B = detail::forwarded_expression_t<Expr2>;
    using V = expression::binary::ExteriorProduct<A, B>;
    return FormBase<V>(std::in_place, std::forward<Expr1>(lhs).expression(),
                       std::forward<Expr2>(rhs).expression());
}

template <concepts::Form Expr>
auto to_basis_coordinates(const Expr &expr) {
    using value_type = typename std::decay_t<Expr>::value_type;
    using extents_type = typename std::decay_t<Expr>::extents_type;
    constexpr rank_type degree = extents_type::rank();

    if constexpr (degree == 0) {
        Vector<value_type, 1> result;
        result(0) = expr();
        return result;
    } else {
        const index_type ambient_dimension = expr.extent(0);
        const index_type basis_size = exterior::detail::binomial(ambient_dimension, degree);
        VectorX<value_type> result(basis_size);
        std::fill(result.begin(), result.end(), value_type{0});
        const auto alt = alternating_part(expr);

        for (index_type idx = 0; idx < basis_size; ++idx) {
            const auto multi_index = exterior::detail::combination_unrank(
                ambient_dimension, degree, idx);
            result(idx) = exterior::detail::evaluate_form(
                alt, multi_index, std::make_index_sequence<degree>{});
        }
        return result;
    }
}

template <concepts::Form Expr>
auto hodge_star(const Expr &expr) {
    using value_type = typename std::decay_t<Expr>::value_type;
    using extents_type = typename std::decay_t<Expr>::extents_type;
    constexpr rank_type degree = extents_type::rank();
    static_assert(degree > 0,
                  "hodge_star currently requires a positive-degree form");

    constexpr index_type ambient_dimension = extents_type::static_extent(0);
    static_assert(ambient_dimension != dynamic_extent,
                  "hodge_star currently requires statically-sized ambient dimension");
    static_assert(degree <= ambient_dimension,
                  "form degree cannot exceed ambient dimension");
    static_assert(
        exterior::detail::static_extents_equal<
            extents_type,
            std::make_index_sequence<(degree > 0 ? degree - 1 : 0)>>::value,
        "hodge_star currently requires equal static extents in every dimension");

    for (rank_type j = 1; j < degree; ++j) {
        ZIPPER_ASSERT(expr.extent(j) == ambient_dimension);
    }

    const auto coords = to_basis_coordinates(expr);
    constexpr rank_type dual_degree = ambient_dimension - degree;
    Vector<value_type, exterior::detail::binomial(ambient_dimension, dual_degree)>
        dual_coords;
    std::fill(dual_coords.begin(), dual_coords.end(), value_type{0});

    const index_type basis_size = coords.extent(0);
    for (index_type idx = 0; idx < basis_size; ++idx) {
        const auto basis_indices =
            exterior::detail::combination_unrank(ambient_dimension, degree, idx);
        const auto complement =
            exterior::detail::complement_indices(ambient_dimension,
                                                 basis_indices);
        const auto dual_index =
            exterior::detail::combination_rank(ambient_dimension, complement);
        const int sign =
            exterior::detail::hodge_sign(ambient_dimension, basis_indices);
        dual_coords(dual_index) = value_type(sign) * coords(idx);
    }

    if constexpr (dual_degree == 0) {
        Form<value_type> result;
        result() = dual_coords(0);
        return result;
    } else {
        return from_basis_coordinates<ambient_dimension>(dual_coords,
                                                         static_rank_t<dual_degree>{});
    }
}

template <index_type AmbientDimension, typename ValueType, index_type... Extents>
auto from_basis_coordinates(const Vector<ValueType, Extents...> &coords,
                            static_rank_t<0>) {
    Form<ValueType> result;
    result() = coords(0);
    return result;
}

template <index_type AmbientDimension, rank_type Degree, typename ValueType,
          index_type... Extents>
auto from_basis_coordinates(const Vector<ValueType, Extents...> &coords,
                            static_rank_t<Degree>)
    -> exterior::detail::static_form_t<ValueType, AmbientDimension, Degree>;

template <index_type AmbientDimension, rank_type Degree, typename ValueType,
          index_type... Extents>
auto from_basis_coordinates(const Vector<ValueType, Extents...> &coords,
                            static_rank_t<Degree>)
    -> exterior::detail::static_form_t<ValueType, AmbientDimension, Degree> {
    static_assert(Degree > 0);
    static_assert(AmbientDimension != dynamic_extent);

    exterior::detail::static_form_t<ValueType, AmbientDimension, Degree> result;
    auto &acc = result.expression().linear_accessor();
    std::fill(acc.begin(), acc.end(), ValueType{0});

    const index_type basis_size = exterior::detail::binomial(AmbientDimension, Degree);
    ZIPPER_ASSERT(coords.extent(0) == basis_size);

    for (index_type idx = 0; idx < basis_size; ++idx) {
        const auto basis_indices =
            exterior::detail::combination_unrank(AmbientDimension, Degree, idx);
        std::vector<index_type> permuted = basis_indices;
        do {
            const int sign = exterior::detail::inversion_parity(permuted);
            exterior::detail::assign_dense_coeff(
                result, permuted, ValueType(sign) * coords(idx),
                std::make_index_sequence<Degree>{});
        } while (std::ranges::next_permutation(permuted).found);
    }

    return result;
}

template <index_type AmbientDimension, rank_type Degree, typename ValueType = double>
auto basis_form(const std::array<index_type, Degree> &indices)
    -> exterior::detail::static_form_t<ValueType, AmbientDimension, Degree> {
    Vector<ValueType, exterior::detail::binomial(AmbientDimension, Degree)> coords;
    std::fill(coords.begin(), coords.end(), ValueType{0});
    coords(exterior::detail::combination_rank(AmbientDimension, indices)) =
        ValueType{1};
    return from_basis_coordinates<AmbientDimension>(coords, static_rank_t<Degree>{});
}

template <concepts::Expression Expr>
auto FormBase<Expr>::operator*() const {
    return hodge_star(*this);
}

} // namespace zipper

#endif
