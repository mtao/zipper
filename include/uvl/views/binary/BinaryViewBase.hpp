#if !defined(UVL_VIEWS_BINARY_BINARYVIEW_HPP)
#define UVL_VIEWS_BINARY_BINARYVIEW_HPP

#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/views/DimensionedViewBase.hpp"

namespace uvl::views::binary {

namespace detail {
template <concepts::ViewDerived ChildA, concepts::ViewDerived ChildB,
          template <typename> typename Base = DimensionedViewBase>
struct DefaultBinaryViewTraits : public views::detail::DefaultViewTraits<> {
    using ATraits = views::detail::ViewTraits<ChildA>;
    using BTraits = views::detail::ViewTraits<ChildB>;
    using lhs_value_type = ATraits::value_type;
    using rhs_value_type = BTraits::value_type;
    // using extents_type = typename BaseTraits::extents_type;
    static_assert(std::is_convertible_v<typename ATraits::value_type,
                                        typename BTraits::value_type> ||
                  std::is_convertible_v<typename BTraits::value_type,
                                        typename ATraits::value_type>);

    // defaulting to first parameter
    using value_type = typename ATraits::value_type;
    constexpr static bool holds_extents = true;
    constexpr static bool is_coefficient_consistent =
        ATraits::is_coefficient_consistent &&
        BTraits::is_coefficient_consistent;
    constexpr static bool is_value_based = true;

    // to pass a base type to the BinaryViewBase
    template <typename Derived>
    using base_type = Base<Derived>;
};
}  // namespace detail

template <typename Derived, concepts::ViewDerived ChildTypeA,
          concepts::ViewDerived ChildTypeB>
class BinaryViewBase
    : public views::detail::ViewTraits<Derived>::template base_type<Derived> {
   public:
    using self_type = BinaryViewBase<Derived, ChildTypeA, ChildTypeB>;
    using traits = uvl::views::detail::ViewTraits<Derived>;
    using extents_type = traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
    using value_type = traits::value_type;
    constexpr static bool holds_extents = traits::holds_extents;
    constexpr static bool is_static = extents_traits::is_static;
    constexpr static bool is_value_based = traits::is_value_based;
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
    using lhs_value_type = traits::lhs_value_type;
    using rhs_value_type = traits::rhs_value_type;

    BinaryViewBase(const BinaryViewBase&) = default;
    BinaryViewBase(BinaryViewBase&&) = default;
    BinaryViewBase& operator=(const BinaryViewBase&) = delete;
    BinaryViewBase& operator=(BinaryViewBase&&) = delete;
    BinaryViewBase(const ChildTypeA& a, const ChildTypeB& b)
        requires(!holds_extents || is_static)
        : m_lhs(a), m_rhs(b) {}

    BinaryViewBase(const ChildTypeA& a, const ChildTypeB& b,
                   const extents_type& e)
        requires(holds_extents && !is_static)
        : m_lhs(a), m_rhs(b), m_extents(e) {}
    using Base =
        views::detail::ViewTraits<Derived>::template base_type<Derived>;
    using Base::extent;

    ChildTypeA& lhs() { return m_lhs; }
    const ChildTypeA& lhs() const { return m_lhs; }

    ChildTypeB& rhs() { return m_rhs; }
    const ChildTypeB& rhs() const { return m_rhs; }

    const extents_type& extents() const
        requires(holds_extents)
    {
        return m_extents;
    }

    value_type get_value(const lhs_value_type& l,
                         const rhs_value_type& r) const {
        return derived().get_value(l, r);
    }

    template <typename... Args>
    value_type coeff(Args&&... args) const
        requires(is_value_based)
    {
        return get_value(m_lhs(std::forward<Args>(args)...),
                         m_rhs(std::forward<Args>(args)...));
    }

   private:
    const ChildTypeA& m_lhs;
    const ChildTypeB& m_rhs;

    std::conditional_t<holds_extents, extents_type, uvl::empty> m_extents;
};

}  // namespace uvl::views::binary
#endif
