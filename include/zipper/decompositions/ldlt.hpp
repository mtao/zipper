#if !defined(ZIPPER_DECOMPOSITIONS_LDLT_HPP)
#define ZIPPER_DECOMPOSITIONS_LDLT_HPP

#include "zipper/concepts/MatrixBaseDerived.hpp"
#include "zipper/utils/extents/is_cubic.hpp"

namespace zipper::decompositions {

template <zipper::concepts::MatrixBaseDerived InputMatType>
    requires(
        zipper::utils::extents::is_cubic<typename InputMatType::extents_type>())
struct LDLT {
   public:
    using value_type = typename InputMatType::value_type;
    using input_extents_type = typename InputMatType::extents_type;
    constexpr static index_type static_extent =
        input_extents_type::static_extent(0) == std::dynamic_extent
            ? input_extents_type::static_extent(1)
            : input_extents_type::static_extent(0);
    using extents_type = zipper::extents<static_extent, static_extent>;
    using MatType = Matrix<value_type, static_extent, static_extent>;

    value_type tripleVectorProduct(auto const& a, auto const& b,
                                   auto const& c) {
        return (a.as_array() * b.as_array()).as_vector().dot(c);
    }
    value_type tripleProduct(const MatType& a, uint i, uint j) {
        return tripleVectorProduct(a.row(i).head(j), a.row(j).head(j),
                                   a.diagonal().head(j));
    }

    void compute(const InputMatType& M) {
        int i, j;
        LD = M;
        for (i = 0; i < M.extent(0); ++i) {
            for (j = 0; j < i; ++j) {
                LD(i, j) -= tripleProduct(LD, i, j);
                LD(i, j) /= LD(j, j);
            }
            LD(i, i) -= tripleProduct(LD, i, i);
        }
    }

    LDLT(const InputMatType& M)
    requires(static_extent != std::dynamic_extent) { compute(M); }
    LDLT(const InputMatType& M)
    requires(static_extent == std::dynamic_extent)
        : MatType(M.extent(0), M.extent(1)) {
        assert(zipper::utils::extents::is_cubic(M.extents()));
        compute(M);
    }

    void solve(auto const& b, auto& x) {
        x = LD.template triangularView<Eigen::UnitLower>().solve(b);

        x = x.as_array() / LD.diagonal().as_array();
        LD.template triangularView<Eigen::UnitLower>().transpose().solveInPlace(
            x);
    }
    Matrix getA() {
        Matrix A = LD.template triangularView<Eigen::UnitLower>().transpose();
        A = LD.diagonal().asDiagonal() * A;
        A = LD.template triangularView<Eigen::UnitLower>() * A;

        return A;
    }

   private:
    MatType LD;
};

template <zipper::concepts::MatrixBaseDerived MatType>
struct LLT {};
}  // namespace zipper::decompositions
