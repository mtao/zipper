// Each *Base.hpp alone provides its deduction guides.
#include <zipper/ArrayBase.hpp>
#include <zipper/DataArrayBase.hpp>
#include <zipper/FormBase.hpp>
#include <zipper/TensorBase.hpp>
#include <zipper/VectorBase.hpp>

#include <array>
#include <span>
#include <type_traits>
#include <vector>

namespace {
using zipper::expression::nullary::MDSpan;
using span_2d = zipper::mdspan<double, zipper::dextents<2>>;
using MDSpan2d = MDSpan<double,
                        zipper::dextents<2>,
                        span_2d::layout_type,
                        span_2d::accessor_type>;
using span_1d = zipper::mdspan<double, zipper::dextents<1>>;
using MDSpan1d = MDSpan<double,
                        zipper::dextents<1>,
                        span_1d::layout_type,
                        span_1d::accessor_type>;

static_assert(std::is_same_v<decltype(zipper::ArrayBase(std::declval<span_2d>())),
                             zipper::ArrayBase<MDSpan2d>>);
static_assert(
    std::is_same_v<decltype(zipper::DataArrayBase(std::declval<span_2d>())),
                   zipper::DataArrayBase<MDSpan2d>>);
static_assert(std::is_same_v<decltype(zipper::FormBase(std::declval<span_1d>())),
                             zipper::FormBase<MDSpan1d>>);
static_assert(
    std::is_same_v<decltype(zipper::TensorBase(std::declval<span_2d>())),
                   zipper::TensorBase<MDSpan2d>>);
static_assert(
    std::is_same_v<decltype(zipper::VectorBase(std::declval<span_1d>())),
                   zipper::VectorBase<MDSpan1d>>);

static_assert(std::is_same_v<
              decltype(zipper::DataArrayBase(std::declval<std::span<double>>())),
              zipper::DataArrayBase<
                  MDSpan<double, zipper::extents<zipper::dynamic_extent>>>>);
static_assert(std::is_same_v<
              decltype(zipper::VectorBase(
                  std::declval<const std::array<double, 3> &>())),
              zipper::VectorBase<MDSpan<const double, zipper::extents<3>>>>);
static_assert(std::is_same_v<
              decltype(zipper::VectorBase(std::declval<std::vector<double> &>())),
              zipper::VectorBase<
                  MDSpan<double, zipper::extents<zipper::dynamic_extent>>>>);
} // namespace

auto main() -> int { return 0; }
