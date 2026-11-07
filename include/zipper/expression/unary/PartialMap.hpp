#if !defined(ZIPPER_EXPRESSION_UNARY_PARTIALMAP_HPP)
#define ZIPPER_EXPRESSION_UNARY_PARTIALMAP_HPP

// PartialMap — applies a function to every sub-tensor of a child.
//
// The sub-tensors span the child's axes In...; the remaining ("kept") axes
// index them.  Fn maps each sub-tensor (a raw expression; wrap it with e.g.
// zipper::MatrixBase(e) or as_array(e) for operations) to a scalar, to an
// expression or zipper object with the same extents for every sub-tensor, or
// to a callable taking the sub-tensor's indices.  The result has the kept axes, in order, plus the axes of
// Fn's result at positions Out... (positions in the result):
//
//   scalar results         reductions: the In axes disappear (Out empty)
//   same-shape results     transforms: Out... = In...
//   other shapes           e.g. (3, W, H) -> (W, H, 4) with In = {0},
//                          Out = {2}
//
// Fn's result extents come from its type when static; otherwise from the
// given extents or, failing that, from Fn applied to the first sub-tensor.
//
// Tensor results declare FiberAssignStrategy, so assignment calls Fn once
// per sub-tensor; coefficient access calls it once per coefficient.

#include "Slice.hpp"
#include "UnaryExpressionBase.hpp"
#include "detail/invert_integer_sequence.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/concepts/Zipper.hpp"
#include "zipper/detail/extents/static_extents_to_array.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/expression/concepts/capabilities.hpp"
#include "zipper/expression/detail/AssignStrategy.hpp"
#include "zipper/utils/extents/assignable_extents.hpp"
#include "zipper/utils/extents/for_each_index.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <type_traits>
#include <utility>

namespace zipper::expression {
namespace unary {
    template <zipper::concepts::QualifiedExpression ExprType,
              typename Fn,
              typename OutAxes,
              rank_type... InAxes>
    class PartialMap;

    namespace detail {
        template <typename T>
        concept TensorResult = zipper::concepts::Zipper<T> || zipper::concepts::Expression<T>;

        // A callable taking a sub-tensor's indices (evaluated lazily),
        // shaped like the sub-tensor.
        template <std::size_t>
        using index_argument = index_type;
        template <typename R, typename Sequence>
        struct index_invoke;
        template <typename R, std::size_t... I>
        struct index_invoke<R, std::index_sequence<I...>> {
            constexpr static bool value = std::is_invocable_v<const R &, index_argument<I>...>;
            // Not ::type: only callables have one.
            using result = std::invoke_result<const R &, index_argument<I>...>;
        };
        template <typename R, typename SubTensor>
        concept IndexableResult =
            !TensorResult<R>
            && index_invoke<R, std::make_index_sequence<SubTensor::extents_type::rank()>>::value;

        // Extents of a scalar, tensor or indexable result.
        template <typename R, typename SubTensor>
        struct result_extents {
            using type = zipper::extents<>;
        };
        template <TensorResult R, typename SubTensor>
        struct result_extents<R, SubTensor> {
            using type = typename R::extents_type;
        };
        template <typename R, typename SubTensor>
            requires IndexableResult<R, SubTensor>
        struct result_extents<R, SubTensor> {
            using type = typename SubTensor::extents_type;
        };

        // Out... as given, or the default: none for scalars, In... for
        // results of the sub-tensors' rank.
        template <typename OutAxes, rank_type ResultRank, rank_type... InAxes>
        struct resolve_out_axes {
            using type = OutAxes;
        };
        template <rank_type ResultRank, rank_type... InAxes>
        struct resolve_out_axes<std::integer_sequence<rank_type>, ResultRank, InAxes...> {
            static_assert(ResultRank == 0 || ResultRank == sizeof...(InAxes),
                          "PartialMap: give the output positions (map<Out...>) of a "
                          "result whose rank differs from the sub-tensors'");
            using type = std::conditional_t<ResultRank == 0,
                                            std::integer_sequence<rank_type>,
                                            std::integer_sequence<rank_type, InAxes...>>;
        };
    } // namespace detail
} // namespace unary

template <zipper::concepts::QualifiedExpression ExprType,
          typename Fn,
          typename OutAxes,
          rank_type... InAxes>
struct detail::ExpressionDetail<unary::PartialMap<ExprType, Fn, OutAxes, InAxes...>> {
    using Base = detail::ExpressionTraits<std::decay_t<ExprType>>;
    using child_extents_type = typename Base::extents_type;
    constexpr static rank_type child_rank = child_extents_type::rank();
    // Maps the kept axes to child axes.
    using index_remover = unary::detail::invert_integer_sequence<child_rank, InAxes...>;
    using kept_extents_type = typename index_remover::template assign_types<
        zipper::extents,
        zipper::detail::extents::static_extents_to_array_v<child_extents_type>>;
    constexpr static rank_type kept_rank = kept_extents_type::rank();

    template <typename>
    struct slice_type_;
    template <rank_type... N>
    struct slice_type_<std::integer_sequence<rank_type, N...>> {
        using type = unary::Slice<const std::decay_t<ExprType> &,
                                  std::conditional_t<!index_remover::in_sequence(N),
                                                     full_extent_t,
                                                     index_type>...>;
    };
    using slice_type =
        typename slice_type_<std::make_integer_sequence<rank_type, child_rank>>::type;
    using sub_tensor_type = slice_type;

    using result_type =
        std::decay_t<std::invoke_result_t<const Fn &, const sub_tensor_type &>>;
    constexpr static bool indexable_result =
        unary::detail::IndexableResult<result_type, sub_tensor_type>;
    constexpr static bool scalar_result =
        !unary::detail::TensorResult<result_type> && !indexable_result;
    using result_extents_type =
        typename unary::detail::result_extents<result_type, sub_tensor_type>::type;
    constexpr static rank_type result_rank = result_extents_type::rank();

    using out_axes = typename unary::detail::
        resolve_out_axes<OutAxes, result_rank, InAxes...>::type;
    constexpr static rank_type rank = kept_rank + result_rank;
    static_assert(out_axes::size() == result_rank,
                  "PartialMap: one output position per axis of Fn's result");

    // Output position of each result axis, and of each kept axis.
    constexpr static auto result_positions = []<rank_type... O>(
                                                 std::integer_sequence<rank_type, O...>) {
        return std::array<rank_type, sizeof...(O)>{{O...}};
    }(out_axes{});
    constexpr static bool is_result_position(rank_type p) {
        for (rank_type j = 0; j < result_rank; ++j) {
            if (result_positions[j] == p) return true;
        }
        return false;
    }
    static_assert([] {
        for (rank_type j = 0; j < result_rank; ++j) {
            if (result_positions[j] >= rank) return false;
            for (rank_type i = 0; i < j; ++i) {
                if (result_positions[i] == result_positions[j]) return false;
            }
        }
        return true;
    }(), "PartialMap: output positions must be distinct and within the result's rank");
    constexpr static auto kept_positions = [] {
        std::array<rank_type, kept_rank> kept{};
        rank_type k = 0;
        for (rank_type p = 0; p < rank; ++p) {
            if (!is_result_position(p)) kept[k++] = p;
        }
        return kept;
    }();

    constexpr static auto static_extents = [] {
        std::array<index_type, rank> e{};
        for (rank_type j = 0; j < result_rank; ++j) {
            e[result_positions[j]] = result_extents_type::static_extent(j);
        }
        for (rank_type k = 0; k < kept_rank; ++k) {
            e[kept_positions[k]] = kept_extents_type::static_extent(k);
        }
        return e;
    }();
    using extents_type = decltype([]<std::size_t... P>(std::index_sequence<P...>) {
        return zipper::extents<static_extents[P]...>{};
    }(std::make_index_sequence<rank>{}));

    using value_type = typename decltype([] {
        if constexpr (scalar_result) {
            return std::type_identity<result_type>{};
        } else if constexpr (indexable_result) {
            return std::type_identity<std::decay_t<typename unary::detail::index_invoke<
                result_type, std::make_index_sequence<result_rank>>::result::type>>{};
        } else {
            return std::type_identity<typename result_type::value_type>{};
        }
    }())::type;
};

template <zipper::concepts::QualifiedExpression ExprType,
          typename Fn,
          typename OutAxes,
          rank_type... InAxes>
struct detail::ExpressionTraits<unary::PartialMap<ExprType, Fn, OutAxes, InAxes...>>
  : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
        ExprType,
        zipper::detail::AccessFeatures::const_value()> {
    using _Detail = detail::ExpressionDetail<unary::PartialMap<ExprType, Fn, OutAxes, InAxes...>>;
    using extents_type = typename _Detail::extents_type;
    using value_type = typename _Detail::value_type;
    using element_type = std::remove_cvref_t<value_type>;

    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
    constexpr static bool has_index_set = false;
    using preferred_layout = std::conditional_t<
        zipper::detail::is_sparse_layout_preference_v<typename _Detail::Base::preferred_layout>,
        zipper::detail::NoLayoutPreference,
        typename _Detail::Base::preferred_layout>;

    // Tensor results are written one sub-tensor at a time.
    using assign_strategy = std::conditional_t<_Detail::scalar_result,
                                               detail::DefaultAssignStrategy,
                                               detail::FiberAssignStrategy<InAxes...>>;
};

namespace unary {
    template <zipper::concepts::QualifiedExpression ExprType,
              typename Fn,
              typename OutAxes,
              rank_type... InAxes>
    class PartialMap
      : public UnaryExpressionBase<PartialMap<ExprType, Fn, OutAxes, InAxes...>, ExprType> {
      public:
        using self_type = PartialMap<ExprType, Fn, OutAxes, InAxes...>;
        using traits = zipper::expression::detail::ExpressionTraits<self_type>;
        using detail_type = zipper::expression::detail::ExpressionDetail<self_type>;
        using extents_type = typename traits::extents_type;
        using value_type = typename traits::value_type;
        using Base = UnaryExpressionBase<self_type, ExprType>;
        using Base::expression;
        using result_extents_type = typename detail_type::result_extents_type;

        constexpr static rank_type rank = detail_type::rank;
        constexpr static rank_type kept_rank = detail_type::kept_rank;
        constexpr static rank_type result_rank = detail_type::result_rank;

        template <typename U, typename F>
            requires std::constructible_from<typename Base::storage_type, U &&>
        PartialMap(U &&expr,
                   F &&fn,
                   std::optional<result_extents_type> result_extents = std::nullopt)
          : Base(std::forward<U>(expr)),
            m_fn(std::forward<F>(fn)),
            m_extents(make_extents(result_extents)) {}

        PartialMap() = delete;
        PartialMap &operator=(const PartialMap &) = delete;
        PartialMap &operator=(PartialMap &&) = delete;
        PartialMap(PartialMap &&) = default;
        PartialMap(const PartialMap &) = default;

        constexpr auto extent(rank_type i) const -> index_type { return m_extents.extent(i); }
        constexpr auto extents() const -> extents_type { return m_extents; }

        template <typename... Args>
        auto coeff(Args &&...idxs) const -> value_type {
            zipper::utils::extents::indices_in_range(this->extents(), idxs...);
            const std::array<index_type, rank> full{{static_cast<index_type>(idxs)...}};
            return apply_at(kept_of(full), [&](const auto &result) -> value_type {
                if constexpr (detail_type::scalar_result) {
                    return result;
                } else {
                    return [&]<rank_type... J>(std::integer_sequence<rank_type, J...>) {
                        return static_cast<value_type>(
                            result(full[detail_type::result_positions[J]]...));
                    }(std::make_integer_sequence<rank_type, result_rank>{});
                }
            });
        }

        template <zipper::concepts::Expression To>
            requires(zipper::expression::concepts::WritableExpression<To>
                     && !std::is_const_v<std::remove_reference_t<To>>
                     && zipper::utils::extents::assignable_extents_v<
                         extents_type,
                         typename To::extents_type>)
        void assign_to(To &to) const {
            const auto kept_extents =
                detail_type::index_remover::get_extents(expression().extents());
            zipper::utils::extents::for_each_index<zipper::storage::layout_right>(
                kept_extents, [&](auto... kept_idxs) {
                    const std::array<index_type, kept_rank> kept{
                        {static_cast<index_type>(kept_idxs)...}};
                    std::array<index_type, rank> full{};
                    for (rank_type k = 0; k < kept_rank; ++k) {
                        full[detail_type::kept_positions[k]] = kept[k];
                    }
                    apply_at(kept, [&](const auto &result) {
                        if constexpr (detail_type::scalar_result) {
                            write(to, full, result);
                        } else {
                            zipper::utils::extents::for_each_index<zipper::storage::layout_right>(
                                result_extents(), [&](auto... result_idxs) {
                                    const std::array<index_type, result_rank> r{
                                        {static_cast<index_type>(result_idxs)...}};
                                    for (rank_type j = 0; j < result_rank; ++j) {
                                        full[detail_type::result_positions[j]] = r[j];
                                    }
                                    write(to, full, result(result_idxs...));
                                });
                        }
                    });
                });
        }

        auto make_owned() const {
            auto owned_child = expression().make_owned();
            return PartialMap<const decltype(owned_child), Fn, typename detail_type::out_axes,
                              InAxes...>(std::move(owned_child), m_fn, result_extents());
        }

      private:
        Fn m_fn;
        extents_type m_extents;

        // Calls `use` with Fn applied to the sub-tensor at kept indices
        // `kept`; the sub-tensor outlives the call, as lazy results may
        // reference it.
        template <typename Use>
        auto apply_at(const std::array<index_type, kept_rank> &kept, Use &&use) const
            -> decltype(auto) {
            return [&]<rank_type... N>(std::integer_sequence<rank_type, N...>) {
                const auto slice_arg = [&]<rank_type A>(std::integral_constant<rank_type, A>) {
                    if constexpr (((A == InAxes) || ...)) {
                        return zipper::full_extent_t{};
                    } else {
                        return kept[detail_type::index_remover::full_rank_to_reduced_indices[A]];
                    }
                };
                const typename detail_type::sub_tensor_type sub(
                    expression(), slice_arg(std::integral_constant<rank_type, N>{})...);
                return use(m_fn(sub));
            }(std::make_integer_sequence<rank_type, detail_type::child_rank>{});
        }

        static auto kept_of(const std::array<index_type, rank> &full)
            -> std::array<index_type, kept_rank> {
            std::array<index_type, kept_rank> kept{};
            for (rank_type k = 0; k < kept_rank; ++k) {
                kept[k] = full[detail_type::kept_positions[k]];
            }
            return kept;
        }

        template <typename To, typename V>
        static void write(To &to, const std::array<index_type, rank> &full, const V &value) {
            [&]<rank_type... P>(std::integer_sequence<rank_type, P...>) {
                to(full[P]...) = value;
            }(std::make_integer_sequence<rank_type, rank>{});
        }

        auto result_extents() const -> result_extents_type {
            return [&]<rank_type... J>(std::integer_sequence<rank_type, J...>) {
                return result_extents_type(m_extents.extent(detail_type::result_positions[J])...);
            }(std::make_integer_sequence<rank_type, result_rank>{});
        }

        auto make_extents(const std::optional<result_extents_type> &given) const -> extents_type {
            const auto kept_extents =
                detail_type::index_remover::get_extents(expression().extents());
            const auto result_extents = [&]() -> result_extents_type {
                if constexpr (result_extents_type::rank_dynamic() == 0) {
                    return result_extents_type{};
                } else {
                    if (given) return *given;
                    if constexpr (detail_type::indexable_result) {
                        return result_extents_type(expression().extent(InAxes)...);
                    } else {
                        for (rank_type k = 0; k < kept_rank; ++k) {
                            ZIPPER_ASSERT(kept_extents.extent(k) > 0);
                        }
                        return apply_at(std::array<index_type, kept_rank>{},
                                        [](const auto &result) -> result_extents_type {
                                            return result.extents();
                                        });
                    }
                }
            }();
            std::array<index_type, rank> e{};
            for (rank_type j = 0; j < result_rank; ++j) {
                e[detail_type::result_positions[j]] = result_extents.extent(j);
            }
            for (rank_type k = 0; k < kept_rank; ++k) {
                e[detail_type::kept_positions[k]] = kept_extents.extent(k);
            }
            return extents_type(e);
        }
    };
} // namespace unary
} // namespace zipper::expression
#endif
