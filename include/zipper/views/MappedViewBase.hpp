

#if !defined(ZIPPER_VIEWS_MAPPEDVIEWBASE_HPP)
#define ZIPPER_VIEWS_MAPPEDVIEWBASE_HPP

#include <spdlog/spdlog.h>

#include "ViewBase.hpp"
#include "detail/ViewTraits.hpp"
#include "zipper/detail//ExtentsTraits.hpp"

namespace zipper::views {

template <typename Derived_>
class MappedViewBase : public ViewBase<Derived_> {
   public:
    using Derived = Derived_;
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
    using Base = ViewBase<Derived>;
    using traits = detail::ViewTraits<Derived>;
    using Base::extent;

    using value_type = traits::value_type;
    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    constexpr static bool IsStatic = extents_traits::is_static;
    using mapping_type = traits::mapping_type;

    const extents_type& extents() const { return m_mapping.extents(); }
    const mapping_type& mapping() const { return m_mapping; }

    MappedViewBase()
        requires(IsStatic)
    {}
    template <typename... Args>
    MappedViewBase(const extents_type& extents) : m_mapping(extents) {}

    template <concepts::ExtentsType E2>
    void resize_extents(const E2& e)
        requires(extents_traits::template is_convertable_from<E2>())
    {
        m_mapping = {e};
    }

   protected:
    template <typename... Indices>
    auto get_index(Indices&&... indices) const -> index_type {
        static_assert((std::is_integral_v<std::decay_t<Indices>> && ...));
        static_assert((!concepts::TupleLike<Indices> && ...));
#if !defined(NDEBUG)
        if constexpr (sizeof...(indices) == 2) {
            spdlog::info("{} {} / {} {}", indices..., extent(0), extent(1));
        }
        if constexpr (sizeof...(indices) ==1 ) {
            spdlog::info("{}  /  {}", indices..., extent(0));
        }
        assert(
            zipper::detail::extents::indices_in_range(extents(), indices...));
#endif
        index_type r = mapping()(std::forward<Indices>(indices)...);
        return r;
    }

   public:
    template <typename... Indices>
    auto coeff(Indices&&... indices) const -> value_type {
        index_type idx = get_index(std::forward<Indices>(indices)...);
        return derived().coeff_linear(idx);
    }
   template <typename... Indices>
   auto coeff_ref(Indices&&... indices)
       -> value_type& requires(traits::is_writable) {
           index_type idx = get_index(std::forward<Indices>(indices)...);
           return derived().coeff_ref_linear(idx);
       }

   template <typename... Indices>
   auto const_coeff_ref(Indices&&... indices) const
       -> const value_type& requires(traits::is_writable) {
           index_type idx = get_index(std::forward<Indices>(indices)...);
           return derived().const_coeff_ref_linear(idx);
       }

   private : mapping_type m_mapping;
};

}  // namespace zipper::views
#endif
