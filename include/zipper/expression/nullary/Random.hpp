#if !defined(ZIPPER_EXPRESSION_NULLARY_RANDOM_HPP)
#define ZIPPER_EXPRESSION_NULLARY_RANDOM_HPP

#include <random>

#include "NullaryExpressionBase.hpp"

namespace zipper::expression {
namespace nullary {
/// Uniform distribution
template <typename T, typename Generator = std::default_random_engine,
          concepts::Extents Extents = zipper::extents<>>
auto uniform_random_view(const Extents &extents = {}, const T &min = 0,
                         const T &max = 1,
                         const Generator &g = Generator{
                             std::random_device{}()});

template <typename T, typename Generator = std::default_random_engine,
          concepts::Extents Extents = zipper::extents<>>
auto normal_random_view(const Extents &extents = {}, const T &mean = 0,
                        const T &stddev = 1,
                        const Generator &g = Generator{std::random_device{}()});
template <typename T, typename Generator = std::default_random_engine,
          index_type... Indices>
auto uniform_random_infinite_view(const T &min = 0, const T &max = 1,
                                  const Generator &g = Generator{
                                      std::random_device{}()});

template <typename T, typename Generator = std::default_random_engine>
auto normal_random_infinite_view(const T &min = 0, const T &max = 1,
                                 const Generator &g = Generator{
                                     std::random_device{}()});

template <template <typename RT> typename Distribution, typename T,
          typename Generator = std::default_random_engine,
          concepts::Extents Extents = zipper::extents<>>
auto random_view(const Extents &extents = {}, const Distribution<T> &d = {},
                 const Generator &g = Generator{std::random_device{}()});
template <template <typename RT> typename Distribution, typename T,
          typename Generator = std::default_random_engine>
auto random_infinite_view(const Distribution<T> &d = {},
                          const Generator &g = Generator{
                              std::random_device{}()});

template <typename Distribution, typename Generator,
          concepts::Extents ExtentsType>
class Random : public NullaryExpressionBase<
                   Random<Distribution, Generator, ExtentsType>>,
               public ExtentsType {
public:
  using self_type = Random<Distribution, Generator, ExtentsType>;
  using nullary_base_type = NullaryExpressionBase<self_type>;

  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = traits::extents_type;
  using extents_traits = traits::extents_traits;
  using value_type = traits::value_type;
  static_assert(!concepts::Extents<value_type>);

  using extents_type::extent;
  using extents_type::extents;
  using extents_type::rank;
  auto extents() const -> const extents_type & { return *this; }

  Random(const Random &) = default;
  Random(Random &&) = default;
  auto operator=(const Random &) -> Random & = default;
  auto operator=(Random &&) -> Random & = default;
  Random(const extents_type &e = {}, const Distribution &d = {},
         const Generator &g = Generator{std::random_device{}()})
      : extents_type(e), m_distribution(d), m_generator(g) {}

  auto get_value() const -> value_type {
    value_type v = m_distribution(m_generator);
    return v;
  }

private:
  mutable Distribution m_distribution = {};
  mutable Generator m_generator = {};
};
template <typename Distribution, typename Generator, concepts::Extents Extents>
Random(const Distribution &, const Generator &, const Extents &)
    -> Random<Distribution, Generator, Extents>;

template <template <typename RT> typename Distribution, typename T,
          typename Generator, concepts::Extents Extents>
auto random_view(const Extents &extents, const Distribution<T> &d,
                 const Generator &g) {
  return Random<Distribution<T>, Generator, Extents>(extents, d, g);
}
template <template <typename RT> typename Distribution, typename T,
          typename Generator>
auto random_infinite_view(const Distribution<T> &d, const Generator &g) {

  return Random<Distribution<T>, Generator, zipper::extents<>>(d, g);
}

template <typename T, typename Generator, concepts::Extents Extents>
auto uniform_random_view(const Extents &extents, const T &min, const T &max,
                         const Generator &g) {
  static_assert(std::is_arithmetic_v<T>);
  if constexpr (std::is_integral_v<T>) {
    return random_view<std::uniform_int_distribution, T, Generator, Extents>(
        extents, std::uniform_int_distribution<T>(min, max), g);
  } else if constexpr (std::is_floating_point_v<T>) {
    return random_view<std::uniform_real_distribution, T, Generator, Extents>(
        extents, std::uniform_real_distribution<T>(min, max), g);
  }
}

template <typename T, typename Generator, concepts::Extents Extents>
auto normal_random_view(const Extents &extents, const T &mean, const T &stddev,
                        const Generator &g) {
  static_assert(std::is_floating_point_v<T>);
  return random_view<T, std::normal_distribution, Generator, Extents>(
      extents, std::normal_distribution<T>(mean, stddev), g);
}

template <typename T, typename Generator>
auto uniform_random_infinite_view(const T &min, const T &max,
                                  const Generator &g) {
  return uniform_random_view<T, Generator>(zipper::extents<>{}, min, max, g);
}

template <typename T, typename Generator>
auto normal_random_infinite_view(const T &mean, const T &stddev,
                                 const Generator &g) {
  return normal_random_view<T, Generator>(zipper::extents<>{}, mean, stddev, g);
}
} // namespace nullary
template <typename Distribution, typename Generator, concepts::Extents Extents>
struct detail::ExpressionTraits<
    nullary::Random<Distribution, Generator, Extents>>
    : public BasicExpressionTraits<
          typename Distribution::result_type, Extents,
          expression::detail::AccessFeatures{
              .is_const = false, .is_reference = false, .is_alias_free = true},
          expression::detail::ShapeFeatures{.is_resizable = true}> {};

} // namespace zipper::expression
#endif
