#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <zipper/COOMatrix.hpp>
#include <zipper/CSMatrix.hpp>
#include <zipper/expression/unary/ExtentView.hpp>

#include "catch_include.hpp"

TEST_CASE("sparse_view_implicit_zero_access", "[sparse][extent_view][ref]") {
    auto check = []<typename Layout>() {
        zipper::COOMatrix<double, 3, 4> coo;
        coo.emplace(0, 1) = 2.0;
        coo.emplace(2, 3) = 5.0;
        coo.compress();
        auto source = coo.template to_cs<Layout>();

        auto check_read = [](const auto &expr) {
            using Traits = zipper::expression::detail::ExpressionTraits<
                std::remove_cvref_t<decltype(expr)>>;
            static_assert(Traits::has_index_set);
            CHECK(expr(0, 0) == 0.0);
            CHECK(expr(1, 2) == 0.0); // Entirely empty row and column.
            CHECK(expr(0, 1) == 2.0);
            CHECK(expr.coeff(0, 0) == 0.0);
            if constexpr (Traits::is_referrable()) {
                CHECK_THROWS_AS(expr.const_coeff_ref(0, 0), std::invalid_argument);
                CHECK(expr.const_coeff_ref(0, 1) == 2.0);
            }
        };
        auto check_mutable = [&](auto &expr) {
            check_read(expr);
            using Traits = zipper::expression::detail::ExpressionTraits<
                std::remove_cvref_t<decltype(expr)>>;
            if constexpr (Traits::is_assignable()) {
                static_assert(std::is_same_v<decltype(expr(0, 1)), double &>);
                CHECK_THROWS_AS(expr(0, 0), std::invalid_argument);
                CHECK_THROWS_AS(expr.coeff_ref(0, 0), std::invalid_argument);
                CHECK(&expr(0, 1) == &source.coeff_ref(0, 1));
                expr(0, 1) = 7.0;
                CHECK(source(0, 1) == 7.0);
                expr(0, 1) = 2.0;
            } else {
                CHECK(expr(0, 0) == 0.0);
            }
        };

        check_mutable(source.expression());
        auto dynamic = zipper::as_dynamic(source.expression());
        check_mutable(dynamic);
        auto retyped = zipper::as_extents<zipper::extents<3, 4>>(dynamic);
        check_mutable(retyped);
        auto mixed = zipper::as_extents<
            zipper::extents<3, zipper::dynamic_extent>>(dynamic);
        check_mutable(mixed);
        auto const_dynamic = zipper::as_dynamic(std::as_const(source).expression());
        check_read(const_dynamic);
        CHECK(const_dynamic(0, 0) == 0.0);

        auto span = source.as_span();
        check_mutable(span.expression());
        auto span_dynamic = zipper::as_dynamic(span.expression());
        check_mutable(span_dynamic);
        auto const_span = source.as_const_span();
        auto const_span_dynamic = zipper::as_dynamic(const_span.expression());
        check_read(const_span_dynamic);
        CHECK(const_span_dynamic(0, 0) == 0.0);

        auto unsafe = source.unsafe();
        check_mutable(unsafe.expression());
        auto unsafe_dynamic = zipper::as_dynamic(unsafe.expression());
        check_mutable(unsafe_dynamic);
        auto ref = source.ref();
        check_mutable(ref.expression());
        auto ref_dynamic = zipper::as_dynamic(ref.expression());
        check_mutable(ref_dynamic);
        auto span_retyped = zipper::as_extents<zipper::extents<3, 4>>(span_dynamic);
        auto wrapped = zipper::MatrixBase(std::move(span_retyped));
        auto wrapped_unsafe = wrapped.unsafe();
        check_mutable(wrapped_unsafe.expression());
        auto wrapped_ref = wrapped.ref();
        check_mutable(wrapped_ref.expression());

        auto transpose = wrapped_ref.transpose();
        CHECK(std::as_const(transpose)(0, 0) == 0.0);
        CHECK(std::as_const(transpose)(2, 1) == 0.0);
        CHECK(transpose(1, 0) == 2.0);
        CHECK(&transpose(1, 0) == &source.coeff_ref(0, 1));
        CHECK_THROWS_AS(transpose(0, 0), std::invalid_argument);
        CHECK_THROWS_AS(transpose.expression().const_coeff_ref(0, 0),
                        std::invalid_argument);
        auto row = wrapped_ref.row(1);
        CHECK(std::as_const(row)(2) == 0.0);
        CHECK(source.compressed_data().nnz() == 2);
    };

    check.template operator()<zipper::storage::layout_right>();
    check.template operator()<zipper::storage::layout_left>();
}

TEST_CASE("sparse_retyped_scaled_transpose_support",
          "[sparse][extent_view][transpose][eval]") {
    auto check = []<typename Layout>() {
        using Preference = zipper::detail::SparseLayoutPreference<Layout>;
        using TransposePreference = zipper::detail::flip_layout_t<Preference>;
        using TransposeLayout = typename TransposePreference::layout_policy;
        zipper::COOMatrix<double, 3, 4> coo;
        coo.emplace(0, 1) = 2.0;
        coo.emplace(0, 3) = 4.0;
        coo.emplace(2, 3) = 5.0;
        coo.compress();
        auto source = coo.template to_cs<Layout>();

        auto dynamic = zipper::as_dynamic(source.expression());
        auto retyped = zipper::as_extents<zipper::extents<3, 4>>(dynamic);
        auto view = zipper::MatrixBase(std::move(retyped));
        auto scaled = 3.0 * view;
        auto check_source_support = [](const auto &expr) {
            using Traits = zipper::expression::detail::ExpressionTraits<
                std::remove_cvref_t<decltype(expr)>>;
            static_assert(Traits::has_index_set);
            static_assert(std::is_same_v<typename Traits::preferred_layout, Preference>);
            CHECK(expr.template index_set<1>(0).contains(1));
            CHECK(expr.template index_set<1>(0).contains(3));
            CHECK_FALSE(expr.template index_set<1>(0).contains(0));
            CHECK(expr.template index_set<1>(1).empty());
            CHECK(expr.template index_set<0>(2).empty());
        };
        check_source_support(dynamic);
        check_source_support(view.expression());
        check_source_support(scaled.expression());

        auto check_transpose = [](const auto &transpose, double scale) {
            using Traits = zipper::expression::detail::ExpressionTraits<
                typename std::remove_cvref_t<decltype(transpose)>::expression_type>;
            static_assert(Traits::has_index_set);
            static_assert(std::is_same_v<typename Traits::preferred_layout,
                                         TransposePreference>);
            const auto &expr = transpose.expression();
            auto rows = expr.template index_set<0>(0);
            auto cols = expr.template index_set<1>(3);
            static_assert(zipper::expression::detail::IndexSet<decltype(rows)>);
            static_assert(std::ranges::forward_range<const decltype(rows)>);
            static_assert(std::ranges::forward_range<const decltype(cols)>);
            CHECK(std::ranges::to<std::vector<zipper::index_type>>(rows)
                  == std::vector<zipper::index_type>{1, 3});
            CHECK(std::ranges::to<std::vector<zipper::index_type>>(cols)
                  == std::vector<zipper::index_type>{0, 2});
            CHECK_FALSE(rows.contains(0));
            CHECK_FALSE(cols.contains(1));
            CHECK(expr.row_range_for_col(1).empty());
            CHECK(expr.col_range_for_row(2).empty());
            CHECK(transpose(0, 0) == 0.0);
            CHECK(transpose(2, 1) == 0.0);
            CHECK(transpose(1, 0) == 2.0 * scale);
            CHECK(transpose(3, 0) == 4.0 * scale);
            CHECK(transpose(3, 2) == 5.0 * scale);

            auto result = transpose.eval();
            static_assert(std::is_same_v<decltype(result),
                zipper::CSMatrix<double, 4, 3, TransposeLayout>>);
            CHECK(result.compressed_data().nnz() == 3);
            CHECK(result(0, 0) == 0.0);
            CHECK(result(1, 0) == 2.0 * scale);
            CHECK(result(3, 0) == 4.0 * scale);
            CHECK(result(3, 2) == 5.0 * scale);
        };

        check_transpose(source.transpose(), 1.0);
        check_transpose(view.transpose(), 1.0);
        check_transpose(scaled.transpose(), 3.0);
        check_transpose((view * 3.0).transpose(), 3.0);
        auto span = source.as_span();
        check_transpose((3.0 * span).transpose(), 3.0);
        auto span_dynamic = zipper::as_dynamic(span.expression());
        auto span_retyped = zipper::as_extents<zipper::extents<3, 4>>(span_dynamic);
        auto span_view = zipper::MatrixBase(std::move(span_retyped));
        check_transpose((3.0 * span_view).transpose(), 3.0);
        check_transpose((span_view.unsafe() * 3.0).transpose(), 3.0);
        check_transpose((span_view.ref() * 3.0).transpose(), 3.0);
        auto evaluated = view.eval();
        static_assert(std::is_same_v<decltype(evaluated),
            zipper::CSMatrix<double, 3, 4, Layout>>);
        CHECK(evaluated.compressed_data().nnz() == 3);
        CHECK(source.compressed_data().nnz() == 3);
    };

    check.template operator()<zipper::storage::layout_right>();
    check.template operator()<zipper::storage::layout_left>();
}
