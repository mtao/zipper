#if !defined(UVL_TENSORBASE_HPP)
#define UVL_TENSORBASE_HPP

#include "UVLBase.hpp"
#include "concepts/TensorBaseDerived.hpp"
#include "uvl/views/binary/TensorProductView.hpp"

namespace uvl {

template <concepts::ViewDerived View>
class TensorBase : public UVLBase<TensorBase, View> {
   public:
    using Base = UVLBase<TensorBase, View>;
    TensorBase() = default;

    using view_type = View;
    using value_type = View::value_type;
    using extents_type = View::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;

    template <index_type... N>
    auto eval(const extents<N...>&) const {
        return Tensor<value_type, N...>(*this);
    }
    auto eval() const { return eval(extents()); }

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
    return TensorBase<views::binary::TensorProductView<
        typename View1::view_type, typename View2::view_type>>(lhs.view(),
                                                               rhs.view());
}
}  // namespace uvl

#endif
