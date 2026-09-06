#include <type_traits>
#include <utility>

#include <zipper/Array.hpp>
#include <zipper/DataArray.hpp>
#include <zipper/Form.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Quaternion.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>

#include "catch_include.hpp"

TEST_CASE("matrix_product_deduction", "[ctad][matrix]") {
    auto check = []<zipper::index_type R, zipper::index_type C>() {
        zipper::Matrix<double, R, 2> a{{1.0, 2.0}, {3.0, 4.0}};
        zipper::Matrix<double, 2, C> b{{2.0, 0.0, 1.0}, {0.0, 3.0, 2.0}};
        zipper::Matrix result = zipper::expression::binary::MatrixProduct(
            a.expression(), b.expression());
        static_assert(std::is_same_v<decltype(result), zipper::Matrix<double, R, C>>);
        a(0, 0) = 100.0;
        CHECK(result.extent(0) == 2);
        CHECK(result.extent(1) == 3);
        CHECK(result(0, 0) == 2.0);
        CHECK(result(0, 1) == 6.0);
        CHECK(result(1, 2) == 11.0);
    };
    check.template operator()<2, 3>();
    check.template operator()<zipper::dynamic_extent, zipper::dynamic_extent>();
    check.template operator()<2, zipper::dynamic_extent>();
    check.template operator()<zipper::dynamic_extent, 3>();
}

TEST_CASE("arbitrary_rank_owner_deduction", "[ctad][dense]") {
    auto check = []<zipper::index_type... N>(zipper::extents<N...> extents) {
        zipper::Array<double, N...> source(extents);
        source = zipper::expression::nullary::Constant<double, N...>(2.0, extents);
        auto sum = source + source;
        auto &raw = sum.expression();

        zipper::Array array = raw;
        zipper::Form form = raw;
        zipper::Tensor tensor = std::as_const(raw);
        zipper::DataArray data = std::as_const(raw);
        static_assert(std::is_same_v<decltype(array), zipper::Array<double, N...>>);
        static_assert(std::is_same_v<decltype(form), zipper::Form<double, N...>>);
        static_assert(std::is_same_v<decltype(tensor), zipper::Tensor<double, N...>>);
        static_assert(std::is_same_v<decltype(data), zipper::DataArray<double, N...>>);

        zipper::Array wrapped_array = sum;
        zipper::Form wrapped_form = zipper::FormBase(raw);
        zipper::Tensor wrapped_tensor = zipper::TensorBase(raw);
        static_assert(std::is_same_v<decltype(wrapped_array), decltype(array)>);
        static_assert(std::is_same_v<decltype(wrapped_form), decltype(form)>);
        static_assert(std::is_same_v<decltype(wrapped_tensor), decltype(tensor)>);

        source = zipper::expression::nullary::Constant<double, N...>(9.0, extents);
        auto check_values = [&](const auto &owner) {
            CHECK(owner.extents() == extents);
            CHECK(zipper::ArrayBase(owner.expression()).sum() ==
                  4.0 * zipper::detail::ExtentsTraits<decltype(extents)>::size(extents));
        };
        check_values(array);
        check_values(form);
        check_values(tensor);
        check_values(data);
        check_values(wrapped_array);
        check_values(wrapped_form);
        check_values(wrapped_tensor);
    };
    check(zipper::extents<>{});
    check(zipper::extents<3>{});
    check(zipper::extents<2, 3>{});
    check(zipper::extents<2, zipper::dynamic_extent, 3>{4});
    check(zipper::extents<zipper::dynamic_extent, zipper::dynamic_extent>{2, 3});
}

TEST_CASE("const_span_expression_deduction", "[ctad][dense]") {
    zipper::Matrix<double, 2, 3> source{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    auto span = source.as_const_span();
    zipper::Matrix matrix = span.expression();
    zipper::Array array = span.expression();
    zipper::Form form = span.expression();
    zipper::Tensor tensor = span.expression();
    zipper::DataArray data = span.expression();
    static_assert(std::is_same_v<decltype(matrix), zipper::Matrix<double, 2, 3>>);
    static_assert(std::is_same_v<decltype(array), zipper::Array<double, 2, 3>>);
    static_assert(std::is_same_v<decltype(form), zipper::Form<double, 2, 3>>);
    static_assert(std::is_same_v<decltype(tensor), zipper::Tensor<double, 2, 3>>);
    static_assert(std::is_same_v<decltype(data), zipper::DataArray<double, 2, 3>>);
    source(0, 0) = 100.0;
    CHECK(matrix(0, 0) == 1.0);
    CHECK(array(0, 0) == 1.0);
    CHECK(form(0, 0) == 1.0);
    CHECK(tensor(0, 0) == 1.0);
    CHECK(data(0, 0) == 1.0);
}

TEST_CASE("existing_vector_quaternion_deduction", "[ctad][dense]") {
    zipper::Vector<double, 4> source{1.0, 2.0, 3.0, 4.0};
    auto sum = source + source;
    zipper::Vector vector = sum.expression();
    zipper::Quaternion quaternion = sum.expression();
    static_assert(std::is_same_v<decltype(vector), zipper::Vector<double, 4>>);
    static_assert(std::is_same_v<decltype(quaternion), zipper::Quaternion<double>>);
    CHECK(vector(0) == 2.0);
    CHECK(quaternion(3) == 8.0);
}

TEST_CASE("reference_backed_form_tensor_construction", "[ctad][dense]") {
    zipper::Form<double, 3> form;
    zipper::Tensor<double, 3> tensor;
    form(0) = 1.0;
    tensor(0) = 2.0;
    zipper::FormBase<const decltype(form)::expression_type &> form_view(form);
    zipper::TensorBase<const decltype(tensor)::expression_type &> tensor_view(tensor);
    CHECK(&form_view.expression() == &form.expression());
    CHECK(&tensor_view.expression() == &tensor.expression());
    form(0) = 3.0;
    tensor(0) = 4.0;
    CHECK(form_view(0) == 3.0);
    CHECK(tensor_view(0) == 4.0);
    auto form_span = form.expression().as_span();
    auto tensor_span = tensor.expression().as_span();
    zipper::FormBase form_ref(form_span);
    zipper::TensorBase tensor_ref(tensor_span);
    zipper::FormBase<decltype(form_span)> form_span_view(form_ref);
    zipper::TensorBase<decltype(tensor_span)> tensor_span_view(tensor_ref);
    form_span_view(0) = 5.0;
    tensor_span_view(0) = 6.0;
    CHECK(form(0) == 5.0);
    CHECK(tensor(0) == 6.0);
}
