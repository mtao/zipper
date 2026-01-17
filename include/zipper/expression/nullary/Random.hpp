#if !defined(ZIPPER_expression_NULLARY_RANDOMVIEW_HPP)
#define ZIPPER_expression_NULLARY_RANDOMVIEW_HPP

#include <random>

#include "NullaryExpressionBase.hpp"

namespace zipper::expression {
namespace nullary {
template <typename Distribution,
          typename Generator = std::default_random_engine,
          index_type... Indices>
class Random;

/// Uniform distribution
template <typename T, typename Generator = std::default_random_engine,
          index_type... Indices>
auto uniform_random_view(const extents<Indices...> &extents = {},
                         const T &min = 0, const T &max = 1,
                         const Generator &g = Generator{
                             std::random_device{}()});

template <typename T, typename Generator = std::default_random_engine,
          index_type... Indices>
auto normal_random_view(const extents<Indices...> &extents = {},
                        const T &mean = 0, const T &stddev = 1,
                        const Generator &g = Generator{std::random_device{}()});
template <typename T, typename Generator = std::default_random_engine,
          index_type... Indices>
auto uniform_random_infinite_view(const T &min = 0, const T &max = 1,
                                  const Generator &g = Generator{
                                      std::random_device{}()});

template <typename T, typename Generator = std::default_random_engine,
          index_type... Indices>
auto normal_random_infinite_view(const T &min = 0, const T &max = 1,
                                 const Generator &g = Generator{
                                     std::random_device{}()});

} // namespace nullary
template <typename Distribution, typename Generator, index_type... Indices>
struct detail::ExpressionTraits<
    nullary::Random<Distribution, Generator, Indices...>>
    : public nullary::detail::DefaultNullaryExpressionTraits<
          typename Distribution::result_type, Indices...> {
  constexpr static bool is_plain_data = false;

  using base_type = SizedExpressionBase<
      typename nullary::Random<Distribution, Generator, Indices...>>;
};

namespace nullary {
template <typename Distribution, typename Generator, index_type... Indices>
class Random
    : public NullaryExpressionBase<Random<Distribution, Generator, Indices...>,
                                   typename Distribution::result_type,
                                   Indices...> {
public:
  using self_type = Random<Distribution, Generator, Indices...>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = traits::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  using value_type = traits::value_type;
  using Base =
      NullaryExpressionBase<Random<Distribution, Generator, Indices...>,
                            typename Distribution::result_type, Indices...>;

  Random(const Random &) = default;
  Random(Random &&) = default;
  Random &operator=(const Random &) = default;
  Random &operator=(Random &&) = default;
  template <typename... Args>
  Random(const Distribution &d = {},
         const Generator &g = Generator{std::random_device{}()}, Args &&...args)
      : Base(std::forward<Args>(args)...), m_distribution(d), m_generator(g) {}
  template <typename... Args>
  Random(const Distribution &d, const unsigned int seed, Args &&...args)
      : Base(std::forward<Args>(args)...), m_distribution(d),
        m_generator(seed) {}

  value_type get_value() const { return m_distribution(m_generator); }

private:
  mutable Distribution m_distribution = {};
  mutable Generator m_generator = {};
};
template <typename Distribution, typename Generator, index_type... Indices>
Random(const Distribution &, const Generator &, const extents<Indices...> &)
    -> Random<Distribution, Generator, Indices...>;

template <typename T, typename Generator, index_type... Indices>
auto uniform_random_view(const extents<Indices...> &extents, const T &min,
                         const T &max, const Generator &g) {
  static_assert(std::is_arithmetic_v<T>);
  if constexpr (std::is_integral_v<T>) {
    return Random<std::uniform_int_distribution<T>, Generator, Indices...>(
        std::uniform_int_distribution<T>{min, max}, g, extents);
  } else if constexpr (std::is_floating_point_v<T>) {
    return Random<std::uniform_real_distribution<T>, Generator, Indices...>(
        std::uniform_real_distribution<T>{min, max}, g, extents);
  }
}

template <typename T, typename Generator, index_type... Indices>
auto normal_random_view(const extents<Indices...> &extents, const T &mean,
                        const T &stddev, const Generator &g) {
  static_assert(std::is_floating_point_v<T>);
  return Random<std::normal_distribution<T>, Generator, Indices...>(
      std::normal_distribution<T>{mean, stddev}, g, extents);
}

template <typename T, typename Generator, index_type... Indices>
auto uniform_random_infinite_view(const T &min, const T &max,
                                  const Generator &g) {
  return uniform_random_view<T, Generator>(zipper::extents<>{}, min, max, g);
}

template <typename T, typename Generator, index_type... Indices>
auto normal_random_infinite_view(const T &min, const T &max,
                                 const Generator &g) {
  return normal_random_view<T, Generator>(zipper::extents<>{}, min, max, g);
}
} // namespace nullary
} // namespace zipper::expression
#endif
