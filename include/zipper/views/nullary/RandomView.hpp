#if !defined(ZIPPER_VIEWS_NULLARY_RANDOMVIEW_HPP)
#define ZIPPER_VIEWS_NULLARY_RANDOMVIEW_HPP

#include <random>

#include "NullaryViewBase.hpp"

namespace zipper::views {
namespace nullary {
template <typename Distribution,
          typename Generator = std::default_random_engine,
          index_type... Indices>
class RandomView;

template <typename T, typename Generator = std::default_random_engine,
          index_type... Indices>
auto uniform_random_view(const extents<Indices...>& extents = {}, const T& min = 0,
                         const T& max = 1,
                         const Generator& g = Generator{
                             std::random_device{}()}) {
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
auto normal_random_view(const extents<Indices...>& extents = {}, const T& mean = 0,
                        const T& stddev = 1,
                        const Generator& g = Generator{
                            std::random_device{}()}) {
    static_assert(std::is_floating_point_v<T>);
    return RandomView<std::normal_distribution<T>, Generator, Indices...>(
        std::normal_distribution<T>{mean, stddev}, g, extents);
}

template <typename T, typename Generator = std::default_random_engine,
          index_type... Indices>
auto uniform_random_infinite_view(const T& min = 0, const T& max = 1,
                                  const Generator& g = Generator{
                                      std::random_device{}()}) {
    return uniform_random_view<T, Generator>(zipper::extents<>{}, min, max, g);
}

template <typename T, typename Generator = std::default_random_engine,
          index_type... Indices>
auto normal_random_infinite_view(const T& min = 0, const T& max = 1,
                                 const Generator& g = Generator{
                                     std::random_device{}()}) {
    return normal_random_view<T, Generator>(zipper::extents<>{}, min, max, g);
}

}  // namespace nullary
template <typename Distribution, typename Generator, index_type... Indices>
struct detail::ViewTraits<
    nullary::RandomView<Distribution, Generator, Indices...>>
    : public nullary::detail::DefaultNullaryViewTraits<
          typename Distribution::result_type, Indices...> {};

namespace nullary {
template <typename Distribution, typename Generator, index_type... Indices>
class RandomView
    : public NullaryViewBase<RandomView<Distribution, Generator, Indices...>,
                             typename Distribution::result_type, Indices...> {
   public:
    using self_type = RandomView<Distribution, Generator, Indices...>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    using value_type = traits::value_type;
    using Base =
        NullaryViewBase<RandomView<Distribution, Generator, Indices...>,
                        typename Distribution::result_type, Indices...>;

    RandomView(const RandomView&) = default;
    RandomView(RandomView&&) = default;
    RandomView& operator=(const RandomView&) = default;
    RandomView& operator=(RandomView&&) = default;
    template <typename... Args>
    RandomView(const Distribution& d = {},
               const Generator& g = Generator{std::random_device{}()}, Args&&... args)
        : Base(std::forward<Args>(args)...),
          m_distribution(d),
          m_generator(g) {}
    template <typename... Args>
    RandomView(const Distribution& d, const unsigned int seed, Args&&... args)
        : Base(std::forward<Args>(args)...),
          m_distribution(d),
          m_generator(seed) {}

    value_type get_value() const { return m_distribution(m_generator); }

   private:
    mutable Distribution m_distribution = {};
    mutable Generator m_generator = {};
};
template <typename Distribution, typename Generator, index_type... Indices>
RandomView(const Distribution&, const Generator&, const extents<Indices...>&)
    -> RandomView<Distribution, Generator, Indices...>;

}  // namespace nullary
}  // namespace zipper::views
#endif
