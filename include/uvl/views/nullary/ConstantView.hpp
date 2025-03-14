
#if !defined(UVL_VIEWS_NULLARY_CONSTANTVIEW_HPP)
#define UVL_VIEWS_NULLARY_CONSTANTVIEW_HPP

#include "NullaryViewBase.hpp"

namespace uvl::views {
namespace nullary {
template <typename T, index_type... Indices>
class ConstantView;

}
template <typename T, index_type... Indices>
struct detail::ViewTraits<nullary::ConstantView<T, Indices...>>
    : public nullary::detail::DefaultNullaryViewTraits<T, Indices...> {};

namespace nullary {
template <typename T, index_type... Indices>
class ConstantView
    : public NullaryViewBase<ConstantView<T, Indices...>, T, Indices...> {
   public:
    using self_type = ConstantView<T, Indices...>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
    using value_type = traits::value_type;
    using Base = NullaryViewBase<ConstantView<T, Indices...>, T, Indices...>;

    template <typename... Args>
    ConstantView(const value_type& v, Args&&... args)
        : Base(std::forward<Args>(args)...), m_value(v) {}

    value_type get_value() const { return m_value; }

   private:
    value_type m_value;
};

template <typename T, index_type... Indices>
ConstantView(const T&, const extents<Indices...>&)
    -> ConstantView<T, Indices...>;

}  // namespace nullary
}  // namespace uvl::views
#endif
