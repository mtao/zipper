#if !defined(UVL_VIEWS_NULLARY_RANDOMVIEW_HPP)
#define UVL_VIEWS_NULLARY_RANDOMVIEW_HPP

#include <random>

#include "uvl/views/DimensionedViewBase.hpp"

namespace uvl::views {
namespace nullary {
template <typename Distribution,
          typename Generator = std::default_random_engine,
          index_type... Indices>
class RandomView;

template <typename T, typename Generator = std::default_random_engine,
          index_type... Indices>
auto uniform_random_view(const extents<Indices...>& extents, const T& min = 0,
                         const T& max = 1, const Generator& g = {}) {
    static_assert(std::is_arithmetic_v<T>);
    if constexpr (std::is_integral_v<T>) {
        return RandomView<std::uniform_int_distribution<T>, Generator,
                          Indices...>(
            std::uniform_int_distribution<T>{min, max}, g, extents);
    } else if constexpr (std::is_floating_point_v<T>) {
        return RandomView<std::uniform_real_distribution<T>, Generator,
                          Indices...>(
            std::uniform_real_distribution<T>{min, max}, g, extents);
    }
}

template <typename T, typename Generator = std::default_random_engine,
          index_type... Indices>
auto normal_random_view(const extents<Indices...>& extents, const T& mean = 0,
                        const T& stddev = 1, const Generator& g = {}) {
    static_assert(std::is_floating_point_v<T>);
    return RandomView<std::normal_distribution<T>, Generator, Indices...>(
        std::normal_distribution<T>{mean, stddev}, g, extents);
}

}  // namespace nullary
template <typename Distribution, typename Generator, index_type... Indices>
struct detail::ViewTraits<
    nullary::RandomView<Distribution, Generator, Indices...>>
//: public nullary::detail::CoeffWiseTraits<A, B> {
//: public detail::ViewTraits<A> {
{
    using extents_type = extents<Indices...>;
    using value_type = Distribution::result_type;
    constexpr static bool is_writable = false;
};

namespace nullary {
template <typename Distribution, typename Generator, index_type... Indices>
class RandomView : public DimensionedViewBase<
                       RandomView<Distribution, Generator, Indices...>> {
   public:
    using self_type = RandomView<Distribution, Generator, Indices...>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
    using value_type = traits::value_type;


    RandomView(const RandomView&) = default;
    RandomView(RandomView&&) = default;
    RandomView& operator=(const RandomView&) = default;
    RandomView& operator=(RandomView&&) = default;
    template <typename... Args>
    RandomView(const Distribution& d = {}, const Generator& g = {},
               Args&&... args)
        : m_distribution(d),
          m_generator(g),
          m_extents(std::forward<Args>(args)...) {}
    template <typename... Args>
    RandomView(const Distribution& d, const unsigned int seed, Args&&... args)
        : m_distribution(d),
          m_generator(seed),
          m_extents(std::forward<Args>(args)...) {}
    using Base = DimensionedViewBase<self_type>;
    using Base::extent;

    constexpr const extents_type& extents() const { return m_extents; }

    template <typename... Args>
    value_type coeff(Args&&...) const {
        return m_distribution(m_generator);
    }

   private:
    mutable Distribution m_distribution;
    mutable Generator m_generator;
    extents_type m_extents;
};  // namespace nullarytemplate<typenameA,typenameB>class AdditionView

template <typename Distribution, typename Generator, index_type... Indices>
RandomView(const Distribution&, const Generator&, const extents<Indices...>&)
    -> RandomView<Distribution, Generator, Indices...>;

}  // namespace nullary
}  // namespace uvl::views
#endif
