#if !defined(UVL_FORMBASE_HPP)
#define UVL_FORMBASE_HPP

#include "UVLBase.hpp"
#include "concepts/FormBaseDerived.hpp"
#include "concepts/TensorBaseDerived.hpp"
#include "detail/extents/static_extents_to_integral_sequence.hpp"
#include "concepts/VectorBaseDerived.hpp"
#include "uvl/views/binary/WedgeProductView.hpp"
#include "uvl/views/binary/FormTensorProductView.hpp"

namespace uvl {

template <concepts::ViewDerived View>
class FormBase : public UVLBase<FormBase, View> {
   public:
    using Base = UVLBase<FormBase, View>;
    FormBase() = default;

    using view_type = View;
    using value_type = View::value_type;
    using extents_type = View::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;

    template <index_type... N>
    auto eval(const std::integer_sequence<index_type,N...>&) const
        requires(std::is_same_v<extents<N...>, extents_type>)
    {
        return Form<value_type, N...>(this->view());
    }
    auto eval() const { return eval(detail::extents::static_extents_to_integral_sequence_t<extents_type>{}); }

    using Base::Base;
    using Base::operator=;
    using Base::cast;
    using Base::swizzle;
    using Base::view;

    FormBase& operator=(concepts::FormBaseDerived auto const& v) {
        view() = v.view();
        return *this;
    }

    template <concepts::FormBaseDerived Other>
    FormBase(const Other& other)
        requires(view_type::is_writable)
        : FormBase(other.view()) {}
    template <concepts::FormBaseDerived Other>
    FormBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        return operator=(other.view());
    }

    template <typename... Slices>
    auto slice() {
        auto v = Base::template slice_view<Slices...>();
        return FormBase<std::decay_t<decltype(v)>>(std::move(v));
    }

    template <typename... Slices>
    auto slice(Slices&&... slices) const {
        auto v = Base::template slice_view<Slices...>(
            std::forward<Slices>(slices)...);
        return FormBase<std::decay_t<decltype(v)>>(std::move(v));
    }
    template <typename... Slices>
    auto slice() const {
        auto v = Base::template slice_view<Slices...>();
        return FormBase<std::decay_t<decltype(v)>>(std::move(v));
    }

    template <typename... Slices>
    auto slice(Slices&&... slices) {
        auto v = Base::template slice_view<Slices...>(
            std::forward<Slices>(slices)...);
        return FormBase<std::decay_t<decltype(v)>>(std::move(v));
    }
};

template <concepts::ViewDerived View>
FormBase(View&& view) -> FormBase<View>;
template <concepts::ViewDerived View>
FormBase(const View& view) -> FormBase<View>;

UNARY_DECLARATION(FormBase, LogicalNot, operator!)
UNARY_DECLARATION(FormBase, BitNot, operator~)
UNARY_DECLARATION(FormBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(FormBase, Plus, operator+)
SCALAR_BINARY_DECLARATION(FormBase, Minus, operator-)
SCALAR_BINARY_DECLARATION(FormBase, Multiplies, operator*)
SCALAR_BINARY_DECLARATION(FormBase, Divides, operator/)

BINARY_DECLARATION(FormBase, Plus, operator+)
BINARY_DECLARATION(FormBase, Minus, operator-)

template <concepts::FormBaseDerived View1, concepts::FormBaseDerived View2>
auto operator*(View1 const& lhs, View2 const& rhs) {
    return FormBase<views::binary::WedgeProductView<
        typename View1::view_type, typename View2::view_type>>(lhs.view(),
                                                               rhs.view());
}

template <concepts::FormBaseDerived View1, concepts::TensorBaseDerived View2>
auto operator*(View1 const& lhs, View2 const& rhs) {
    return FormBase<views::binary::FormTensorProductView<
        typename View1::view_type, typename View2::view_type>>(lhs.view(),
                                                               rhs.view());
}
template <concepts::FormBaseDerived View1, concepts::VectorBaseDerived View2>
auto operator*(View1 const& lhs, View2 const& rhs) {
    using TV = views::binary::TensorProductView<typename View1::view_type, typename View2::view_type>;
    using R = views::unary::PartialTraceView<
TV, View1::extents_type::rank()-1, View1::extents_type::rank()>;

    TV t(lhs.view(), rhs.view());
    R r(t);
    return FormBase<R>(t);
    //unary::PartialTraceView<binary::TensorProductView<A,B>> m_trace;
    //return FormBase<views::binary::FormTensorProductView<
    //    typename View1::view_type, typename View2::view_type>>(lhs.view(),
    //                                                           rhs.view());
}

}  // namespace uvl

#include "Form.hpp"
#endif
