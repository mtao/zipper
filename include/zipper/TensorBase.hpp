#if !defined(ZIPPER_TENSORBASE_HPP)
#define ZIPPER_TENSORBASE_HPP

#include "ZipperBase.hpp"
#include "concepts/TensorBaseDerived.hpp"
#include "detail/extents/static_extents_to_integral_sequence.hpp"
#include "zipper/views/binary/TensorProductView.hpp"

namespace zipper {
//    namespace detail {
//        template <typename>
//        struct tensor_extent_utils;
//        template <index_type... Idxs>
//        struct tensor_extent_utils<zipper::extents<Idxs...>> {
//            template <typename T>
//                using plain_object_storage = storage::PlainObjectStorage<T,
//
//        };
//        }

template <concepts::ViewDerived View>
class TensorBase : public ZipperBase<TensorBase, View> {
   public:
    using Base = ZipperBase<TensorBase, View>;
    TensorBase() = default;

    using view_type = View;
    using value_type = View::value_type;
    using extents_type = View::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;

    template <index_type... N>
    auto eval(const std::integer_sequence<index_type, N...>&) const
        requires(std::is_same_v<extents<N...>, extents_type>)
    {
        return Tensor_<value_type, extents<N...>>(this->view());
    }
    auto eval() const {
        return eval(detail::extents::static_extents_to_integral_sequence_t<
                    extents_type>{});
    }

    using Base::Base;
    using Base::operator=;
    using Base::cast;
    using Base::swizzle;
    using Base::view;

    TensorBase& operator=(concepts::TensorBaseDerived auto const& v) {
        view() = v.view();
        return *this;
    }

    template <concepts::TensorBaseDerived Other>
    TensorBase(const Other& other)
        requires(view_type::is_writable)
        : TensorBase(other.view()) {}
    template <concepts::TensorBaseDerived Other>
    TensorBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        return operator=(other.view());
    }
    TensorBase& operator=(concepts::TensorBaseDerived auto && v) {
        return Base::operator=(v.view());
    }

    template <typename... Slices>
    auto slice() {
        auto v = Base::template slice_view<Slices...>();
        return TensorBase<std::decay_t<decltype(v)>>(std::move(v));
    }

    template <typename... Slices>
    auto slice(Slices&&... slices) const {
        auto v = Base::template slice_view<Slices...>(
            std::forward<Slices>(slices)...);
        return TensorBase<std::decay_t<decltype(v)>>(std::move(v));
    }
    template <typename... Slices>
    auto slice() const {
        auto v = Base::template slice_view<Slices...>();
        return TensorBase<std::decay_t<decltype(v)>>(std::move(v));
    }

    template <typename... Slices>
    auto slice(Slices&&... slices) {
        auto v = Base::template slice_view<Slices...>(
            std::forward<Slices>(slices)...);
        return TensorBase<std::decay_t<decltype(v)>>(std::move(v));
    }
};

template <concepts::ViewDerived View>
TensorBase(View&& view) -> TensorBase<View>;
template <concepts::ViewDerived View>
TensorBase(const View& view) -> TensorBase<View>;

template <class T, std::size_t Size = std::dynamic_extent>
TensorBase(const std::span<T, Size>& s)
    -> TensorBase<storage::SpanStorage<T, zipper::extents<Size>>>;
template <class T, std::size_t Size = std::dynamic_extent>
TensorBase(std::span<const T, Size>& s)
    -> TensorBase<storage::SpanStorage<const T, zipper::extents<Size>>>;

UNARY_DECLARATION(TensorBase, LogicalNot, operator!)
UNARY_DECLARATION(TensorBase, BitNot, operator~)
UNARY_DECLARATION(TensorBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(TensorBase, Plus, operator+)
SCALAR_BINARY_DECLARATION(TensorBase, Minus, operator-)
SCALAR_BINARY_DECLARATION(TensorBase, Multiplies, operator*)
SCALAR_BINARY_DECLARATION(TensorBase, Divides, operator/)

BINARY_DECLARATION(TensorBase, Plus, operator+)
BINARY_DECLARATION(TensorBase, Minus, operator-)

template <concepts::TensorBaseDerived View1, concepts::TensorBaseDerived View2>
auto operator*(View1 const& lhs, View2 const& rhs) {
    using V = views::binary::TensorProductView<typename View1::view_type,
                                               typename View2::view_type>;
    return TensorBase<V>(V(lhs.view(), rhs.view()));
}
}  // namespace zipper

#endif
