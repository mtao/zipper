#include "catch_include.hpp"

#include <zipper/Array.hpp>
#include <zipper/DataArray.hpp>
#include <zipper/Form.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Quaternion.hpp>
#include <zipper/Ref.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/transform/Scaling.hpp>

#include <type_traits>
#include <utility>

namespace {
using namespace zipper;
using V = Vector<double, 4>;
using M = Matrix<double, 2, 2>;
using E = V::expression_type;
using ME = M::expression_type;
using Borrowed = VectorBase<E &>;
using ReadOnly = VectorBase<const E &>;

template <typename T>
concept WritableVector = requires(T &v) { v(0) = 1.0; };
template <typename T>
concept WritableMatrix = requires(T &v) { v(0, 0) = 1.0; };
template <typename T, typename R>
concept Assignable = requires(T &v, const R &rhs) { v = rhs; };
template <typename T, typename R>
concept AddAssignable = requires(T &v, const R &rhs) { v += rhs; };
template <typename T>
concept ScaleAssignable = requires(T &v) { v *= 2.0; v /= 2.0; };
template <typename T>
concept Normalizable = requires(T &v) { v.normalize(); };
template <typename T>
concept Resizable = requires(T &v) { v.resize(4); };
template <typename T>
concept Fillable = requires(T &v) { v.fill(1.0); };
template <typename T>
concept InitializerAssignable = requires(T &v) { v = {1.0, 2.0, 3.0, 4.0}; };
template <typename T>
concept HasRef = requires(T &&v) { std::forward<T>(v).ref(); };

static_assert(detail::AccessFeatures::from_type<const E &>().is_const);
static_assert(detail::AccessFeatures::from_type<const E &&>().is_const);
static_assert(!detail::AccessFeatures::from_type<E &>().is_const);
static_assert(detail::AccessFeatures::from_type<const E &>().is_reference);
static_assert(!detail::AccessFeatures::from_type<const E &>().is_assignable());
// Intrinsic capabilities intentionally do not answer qualified-object questions.
static_assert(expression::concepts::WritableExpression<const E &>);
static_assert(ReadOnly::is_const);
static_assert(!Borrowed::is_const);
static_assert(std::is_same_v<decltype(std::declval<Borrowed &>().expression()), E &>);
static_assert(std::is_same_v<decltype(std::declval<Borrowed &&>().expression()), E &>);
static_assert(std::is_same_v<decltype(std::declval<const Borrowed &>().expression()), const E &>);
static_assert(std::is_same_v<decltype(std::declval<const Borrowed &&>().expression()), const E &>);
static_assert(std::is_same_v<decltype(std::declval<const V &&>().expression()), const E &&>);
static_assert(std::is_same_v<decltype(std::declval<ReadOnly &&>().expression()), const E &>);
static_assert(WritableVector<Borrowed>);
static_assert(!WritableVector<const Borrowed>);
static_assert(!WritableVector<ReadOnly>);
static_assert(!Assignable<ReadOnly, V>);
static_assert(!Assignable<ReadOnly, E>);
static_assert(!AddAssignable<ReadOnly, V>);
static_assert(!ScaleAssignable<ReadOnly>);
static_assert(!Normalizable<ReadOnly>);
static_assert(!InitializerAssignable<ReadOnly>);
static_assert(!Resizable<VectorBase<const VectorX<double>::expression_type &>>);
static_assert(!Assignable<MatrixBase<const ME &>, M>);
static_assert(!Assignable<MatrixBase<const ME &>, ME>);
static_assert(!WritableMatrix<MatrixBase<const ME &>>);
static_assert(!Assignable<ArrayBase<const E &>, Array<double, 4>>);
static_assert(!AddAssignable<ArrayBase<const E &>, Array<double, 4>>);
static_assert(!ScaleAssignable<ArrayBase<const E &>>);
static_assert(!Assignable<FormBase<const E &>, Form<double, 4>>);
static_assert(!Assignable<TensorBase<const E &>, Tensor<double, 4>>);
static_assert(!Assignable<DataArrayBase<const E &>, DataArray<double, 4>>);
static_assert(!Fillable<DataArrayBase<const E &>>);
static_assert(!Assignable<QuaternionBase<const E &>, Quaternion<double>>);
static_assert(!Normalizable<QuaternionBase<const E &>>);
static_assert(HasRef<V &> && HasRef<const V &> && HasRef<ReadOnly &>);
static_assert(!HasRef<V> && !HasRef<const V>);
static_assert(!HasRef<Borrowed> && !HasRef<const Borrowed>);
static_assert(std::is_same_v<decltype(std::declval<VectorBase<E &&> &&>().expression()), E &>);
static_assert(std::is_same_v<decltype(std::declval<const VectorBase<E &&> &&>().expression()), const E &>);
static_assert(std::is_same_v<detail::member_child_storage_t<Borrowed, E &>, E &>);
static_assert(std::is_same_v<detail::member_child_storage_t<const Borrowed, E &>, const E &>);
static_assert(std::is_same_v<detail::member_child_storage_t<V, const E>, const E>);
static_assert(std::is_same_v<detail::member_child_storage_t<const V, E>, const E>);
static_assert(std::is_same_v<detail::member_child_storage_t<V, E &&>, E &>);
static_assert(!Assignable<decltype(std::declval<const Form<double, 4> &>().slice(full_extent_t{})), Form<double, 4>>);
static_assert(!Assignable<decltype(std::declval<const Tensor<double, 4> &>().slice(full_extent_t{})), Tensor<double, 4>>);

TEST_CASE("qualification_borrowed_temporary_named_views", "[qualification][views]") {
    V v{1.0, 2.0, 3.0, 4.0};
    auto head = as_vector(v).head<2>();
    auto segment = as_vector(v).segment<1, 2>();
    auto dynamic_segment = as_vector(v).segment(2, 2);
    auto tail = as_vector(v).tail(2);
    static_assert(decltype(head)::stores_references);
    static_assert(!std::is_copy_constructible_v<decltype(head)>);
    head(0) = 11.0;
    segment(0) = 12.0;
    dynamic_segment(0) = 13.0;
    tail(1) = 14.0;
    CHECK(v == V{11.0, 12.0, 13.0, 14.0});

    M m{{1.0, 2.0}, {3.0, 4.0}};
    auto transpose = as_matrix(m).transpose();
    auto slice = as_matrix(m).slice(full_extent_t{}, full_extent_t{});
    auto column = as_matrix(m).col(1);
    transpose(0, 1) = 31.0;
    slice(0, 0) = 10.0;
    column(1) = 41.0;
    CHECK(m(1, 0) == 31.0);
    CHECK(m(0, 0) == 10.0);
    CHECK(m(1, 1) == 41.0);

    auto array_slice = as_array(v).slice(full_extent_t{});
    auto form_slice = as_form(v).slice(full_extent_t{});
    auto tensor_slice = as_tensor(v).slice(full_extent_t{});
    array_slice(0) = 21.0;
    form_slice(1) = 22.0;
    tensor_slice(2) = 23.0;
    CHECK(v(0) == 21.0);
    CHECK(v(1) == 22.0);
    CHECK(v(2) == 23.0);

    VectorX<double> dynamic{1.0, 2.0, 3.0, 4.0};
    auto dynamic_head = as_vector(dynamic).head<2>();
    CHECK(&dynamic_head(0) == &dynamic(0));
    auto rvalue_reference = VectorBase<E &&>(std::in_place, std::move(v.expression()));
    auto reference_head = std::move(rvalue_reference).head<2>();
    reference_head(0) = 25.0;
    CHECK(v(0) == 25.0);
    auto reference_sum = std::move(rvalue_reference) + as_vector(v);
    v(0) = 26.0;
    CHECK(reference_sum(0) == 52.0);
}

TEST_CASE("qualification_conversions_preserve_referents", "[qualification][as]") {
    V v{1.0, 2.0, 3.0, 4.0};
    auto converted = as_vector(as_form(as_array(v)));
    converted(0) = 10.0;
    auto quaternion = as_quaternion(as_array(v));
    quaternion.w() = 11.0;
    CHECK(v(0) == 11.0);
    auto read_only = as_vector(std::as_const(v));
    auto array = as_array(read_only);
    auto vector = as_vector(std::move(read_only));
    static_assert(!WritableVector<decltype(array)>);
    static_assert(!WritableVector<decltype(vector)>);
    static_assert(!WritableVector<decltype(read_only.ref())>);
    static_assert(!WritableVector<decltype(read_only.unsafe())>);
    CHECK(vector(0) == 11.0);
    auto unsafe_borrow = as_vector(v).unsafe();
    unsafe_borrow(0) = 12.0;
    CHECK(v(0) == 12.0);

    Matrix<double, 4, 1> column{{1.0}, {2.0}, {3.0}, {4.0}};
    auto squeezed = as_vector(as_matrix(column));
    squeezed(2) = 30.0;
    CHECK(column(2, 0) == 30.0);
    auto const_squeezed = as_quaternion(as_matrix(std::as_const(column)));
    static_assert(!WritableVector<decltype(const_squeezed)>);
    CHECK(const_squeezed.y() == 30.0);
}

TEST_CASE("qualification_const_rvalues_stay_read_only", "[qualification][const]") {
    const V v{1.0, 2.0, 3.0, 4.0};
    const M m{{1.0, 2.0}, {3.0, 4.0}};
    auto head = std::move(v).head<2>();
    auto segment = std::move(v).segment<1, 2>();
    auto transpose = std::move(m).transpose();
    auto slice = std::move(m).slice(full_extent_t{}, full_extent_t{});
    auto array = as_array(std::move(v));
    auto matrix = as_matrix(std::move(m));
    auto unsafe = std::move(v).unsafe();
    static_assert(!WritableVector<decltype(head)>);
    static_assert(!WritableVector<decltype(segment)>);
    static_assert(!WritableMatrix<decltype(transpose)>);
    static_assert(!WritableMatrix<decltype(slice)>);
    static_assert(!WritableVector<decltype(array)>);
    static_assert(!WritableMatrix<decltype(matrix)>);
    static_assert(!WritableVector<decltype(unsafe)>);
    static_assert(!decltype(head)::stores_references);
    CHECK(head(1) == 2.0);
    CHECK(segment(0) == 2.0);
    CHECK(transpose(1, 0) == 2.0);
    CHECK(slice(1, 0) == 3.0);
    CHECK(array(3) == 4.0);
    CHECK(matrix(1, 1) == 4.0);
    CHECK(unsafe(2) == 3.0);

    V mutable_v{1.0, 2.0, 3.0, 4.0};
    const auto borrowed = as_vector(mutable_v);
    auto borrowed_head = std::move(borrowed).head<2>();
    auto borrowed_array = as_array(std::move(borrowed));
    static_assert(!WritableVector<decltype(borrowed_head)>);
    static_assert(!WritableVector<decltype(borrowed_array)>);
    static_assert(decltype(borrowed_head)::stores_references);
    mutable_v(0) = 9.0;
    CHECK(borrowed_head(0) == 9.0);
    CHECK(borrowed_array(0) == 9.0);
}

TEST_CASE("qualification_owned_temporaries_and_ref_returnability", "[qualification][ref]") {
    auto head = V{1.0, 2.0, 3.0, 4.0}.head<2>();
    auto transpose = M{{1.0, 2.0}, {3.0, 4.0}}.transpose();
    auto converted = as_vector(as_form(V{1.0, 2.0, 3.0, 4.0}));
    static_assert(!decltype(head)::stores_references);
    static_assert(!decltype(transpose)::stores_references);
    static_assert(!decltype(converted)::stores_references);
    CHECK(head(1) == 2.0);
    CHECK(transpose(1, 0) == 2.0);
    CHECK(converted(3) == 4.0);

    V v{1.0, 2.0, 3.0, 4.0};
    auto return_head = [&v] {
        auto r = v.ref();
        auto result = r.head<2>();
        return result;
    };
    auto returned = return_head();
    static_assert(std::is_copy_constructible_v<decltype(returned)>);
    returned(0) = 5.0;
    CHECK(v(0) == 5.0);
    auto return_conversion = [&v] {
        auto r = v.ref();
        auto result = as_form(r);
        return result;
    };
    auto form = return_conversion();
    static_assert(!decltype(form)::stores_references);
    form(1) = 6.0;
    CHECK(v(1) == 6.0);

    const auto r = v.ref();
    auto const_head = r.head<2>();
    auto const_rvalue_head = std::move(r).head<2>();
    static_assert(std::is_copy_constructible_v<decltype(const_head)>);
    static_assert(std::is_copy_constructible_v<decltype(const_rvalue_head)>);
    static_assert(!WritableVector<decltype(const_head)>);
    static_assert(!WritableVector<decltype(const_rvalue_head)>);
    CHECK(const_head(1) == 6.0);
    CHECK(const_rvalue_head(0) == 5.0);
}

TEST_CASE("qualification_operators_borrow_and_own", "[qualification][operators]") {
    V v{1.0, 2.0, 3.0, 4.0};
    auto sum = as_vector(v) + as_vector(v);
    auto scaled = 2.0 * as_vector(v);
    M m{{1.0, 0.0}, {0.0, 1.0}};
    auto product = as_matrix(m) * as_matrix(m);
    v(0) = 5.0;
    m(0, 0) = 3.0;
    CHECK(sum(0) == 10.0);
    CHECK(scaled(0) == 10.0);
    CHECK(product(0, 0) == 9.0);

    const V cv{1.0, 2.0, 3.0, 4.0};
    auto const_sum = std::move(cv) + V{1.0, 1.0, 1.0, 1.0};
    auto const_scaled = std::move(cv) * 2.0;
    auto negated = -std::move(cv);
    static_assert(!decltype(const_sum)::stores_references);
    static_assert(!decltype(const_scaled)::stores_references);
    CHECK(const_sum(0) == 2.0);
    CHECK(const_scaled(3) == 8.0);
    CHECK(negated(1) == -2.0);

    auto return_product = [&m] {
        auto r = m.ref();
        auto result = r * r;
        return result;
    };
    auto ref_product = return_product();
    static_assert(!decltype(ref_product)::stores_references);
    CHECK(ref_product(0, 0) == 9.0);

    Quaternion<double> q{1.0, 0.0, 0.0, 0.0};
    auto return_quaternion_product = [&q] {
        auto r = q.ref();
        auto result = r * r;
        return result;
    };
    auto quaternion_product = return_quaternion_product();
    static_assert(!decltype(quaternion_product)::stores_references);
    CHECK(quaternion_product.w() == 1.0);

    const transform::Scaling<double, 2> scaling(2.0);
    auto linear = std::move(scaling).linear();
    static_assert(!WritableMatrix<decltype(linear)>);
    CHECK(linear(0, 0) == 2.0);
}
} // namespace
