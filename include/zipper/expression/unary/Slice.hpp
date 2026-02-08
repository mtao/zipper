#if !defined(ZIPPER_EXPRESSION_UNARY_SLICE_HPP)
#define ZIPPER_EXPRESSION_UNARY_SLICE_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/IndexSlice.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/concepts/Index.hpp"
#include "zipper/concepts/stl.hpp"
#include "zipper/detail/constexpr_arithmetic.hpp"
#include "zipper/detail/is_integral_constant.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"

namespace zipper::expression {
namespace unary {
template <zipper::concepts::Expression ExprType,
          typename... Slices>
class Slice;

namespace _detail_slice {
template <typename T>
struct slice_helper;

template <typename OffsetType, typename ExtentType, typename StrideType>
struct slice_helper<strided_slice<OffsetType, ExtentType, StrideType>> {
   public:
    using type = strided_slice<OffsetType, ExtentType, StrideType>;
    constexpr slice_helper(const type &t) : m_slice(t) {}
    constexpr static auto get_extent(const auto &stride, const auto &extent) {
        return extent > 0 ? 1 + (extent - 1) / stride : 0;
    }

    template <rank_type N, zipper::concepts::Extents ET>
    constexpr static index_type static_extent() {
        if constexpr (zipper::detail::is_integral_constant_v<StrideType> &&
                      zipper::detail::is_integral_constant_v<ExtentType>) {
            return get_extent(StrideType::value, ExtentType::value);
        } else {
            return std::dynamic_extent;
        }
    }

    template <rank_type N, zipper::concepts::Extents ET>
    static auto internal_extent(const type &t, const ET &e) {
        if constexpr (std::is_same_v<ExtentType,
                                     static_index_t<std::dynamic_extent>>) {
            return e.extent(N);
        } else {
            return t.extent;
        }
    }

    template <rank_type N, zipper::concepts::Extents ET>
    constexpr static index_type extent(const type &t, const ET &ex) {
        zipper::detail::ConstexprArithmetic s = t.stride;
        zipper::detail::ConstexprArithmetic e = internal_extent<N>(t, ex);
        return get_extent(s, e);
    }

    index_type get_index(index_type input) const {
        index_type start = zipper::detail::first_of(m_slice);
        index_type stride = zipper::detail::stride_of(m_slice);
        return start + input * stride;
    }

   private:
    type m_slice;
};
template <>
struct slice_helper<full_extent_t> {
   public:
    using type = full_extent_t;
    constexpr slice_helper(const type &) {}
    template <rank_type N, zipper::concepts::Extents ET>
    constexpr static index_type static_extent() {
        return ET::static_extent(N);
    }
    template <rank_type N, zipper::concepts::Extents ET>
    constexpr static index_type extent(const type &, const ET &e) {
        return e.extent(N);
    }

    constexpr index_type get_index(index_type input) const { return input; }
};
template <typename T>
    requires(std::is_integral_v<T>)
struct slice_helper<T> {
   public:
    using type = index_type;
    constexpr slice_helper(const type &t) : m_slice(t) {}
    template <rank_type N, zipper::concepts::Extents ET>
    constexpr static index_type static_extent() {
        return 1;
    }
    template <rank_type N, zipper::concepts::Extents ET>
    constexpr static index_type extent(const type & /*t*/, const ET &) {
        return static_extent<N, ET>();
    }

    constexpr index_type get_index() const { return m_slice; }

   private:
    index_type m_slice;
};

template <index_type N>
struct slice_helper<std::integral_constant<index_type, N>> {
   public:
    using type = std::integral_constant<index_type, N>;
    constexpr slice_helper(std::integral_constant<index_type, N>) {}
    template <rank_type M, zipper::concepts::Extents ET>
    constexpr static index_type static_extent() {
        return 1;
    }
    template <rank_type M, zipper::concepts::Extents ET>
    constexpr static index_type extent(const type &, const ET &) {
        return static_extent<M, ET>();
    }

    constexpr index_type get_index() const { return N; }
};
template <zipper::concepts::QualifiedExpression Expr>
    requires(Expr::extents_type::rank() == 1)
struct slice_helper<Expr> {
   public:
    using type = Expr;
    constexpr slice_helper(const type &t) : m_slice(t) {}
    template <rank_type M, zipper::concepts::Extents ET>
    constexpr static index_type static_extent() {
        return Expr::extents_type::static_extent(0);
    }
    template <rank_type M, zipper::concepts::Extents ET>
    constexpr static index_type extent(const type &t, const ET &) {
        return t.extent(0);
    }

    constexpr index_type get_index(index_type input) const {
        return m_slice(input);
    }

   private:
    type m_slice;
};

template <size_t N>
struct slice_helper<std::array<index_type, N>> {
   public:
    using type = std::array<index_type, N>;
    constexpr slice_helper(const type &t) : m_slice(t) {}
    template <rank_type M, zipper::concepts::Extents ET>
    constexpr static index_type static_extent() {
        return index_type(N);
    }
    template <rank_type M, zipper::concepts::Extents ET>
    constexpr static index_type extent(const type & /*t*/, const ET &) {
        return static_extent<M, ET>();
    }

    constexpr index_type get_index(index_type input) const {
        return m_slice[input];
    }

   private:
    type m_slice;
};
template <>
struct slice_helper<std::vector<index_type>> {
   public:
    using type = std::vector<index_type>;
    constexpr slice_helper(const type &t) : m_slice(t) {}
    template <rank_type N, zipper::concepts::Extents ET>
    constexpr static index_type static_extent() {
        return std::dynamic_extent;
    }
    template <rank_type N, zipper::concepts::Extents ET>
    constexpr static index_type extent(const type &t, const ET &) {
        return t.size();
    }

    constexpr index_type get_index(index_type input) const {
        return m_slice[input];
    }

   private:
    std::vector<index_type> m_slice;
};

template <zipper::concepts::Extents, typename, typename...>
struct _slice_extent_helper;
template <zipper::concepts::Extents ET, rank_type... N, typename... Slices>
struct _slice_extent_helper<ET, std::integer_sequence<rank_type, N...>,
                            Slices...> {
    template <rank_type R>
    using get_slice_t =
        std::decay_t<std::tuple_element_t<R, std::tuple<Slices...>>>;

    template <size_t M>
    using single_slice_helper = slice_helper<get_slice_t<M>>;

    template <std::size_t... Indices>
    consteval static std::array<rank_type, sizeof...(Indices)> get_actionable(
        std::index_sequence<Indices...>)
        requires(sizeof...(Indices) == ET::rank())
    {
        constexpr size_t Rank = sizeof...(Indices);
        using tuple_type = std::tuple<Slices...>;
        constexpr auto eval =
            []<rank_type J>(std::integral_constant<rank_type, J>) -> rank_type {
            if (zipper::concepts::IndexSlice<
                    std::tuple_element_t<J, tuple_type>> &&
                !zipper::concepts::Index<
                    std::tuple_element_t<J, tuple_type>>) {
                return 0;
            } else {
                return std::dynamic_extent;
            }
        };
        std::array<rank_type, Rank> r{
            {eval(std::integral_constant<size_t, Indices>{})...}};
        rank_type index = 0;
        for (auto &v : r) {
            if (v == 0) {
                v = index++;
            }
        }
        return r;
    }

    constexpr static auto actionable_indices =
        get_actionable(std::make_index_sequence<ET::rank()>{});
    constexpr static rank_type rank = []() {
        rank_type j = 0;
        for (const auto &i : actionable_indices) {
            if (i != std::dynamic_extent) {
                j++;
            }
        }
        return j;
    }();
    template <typename>
    struct _extents_maker;

    template <rank_type... Rs>
    struct _extents_maker<std::integer_sequence<rank_type, Rs...>> {
        consteval static rank_type get_rank(rank_type r) {
            for (rank_type j = 0; j < actionable_indices.size(); ++j) {
                if (actionable_indices[j] == r) {
                    return j;
                }
            }
            // clang doesn't catch that all executions of this consteval turn
            // out to return
            assert(false);
            return rank + 1;
        }
        consteval static rank_type get_dynamic_rank(rank_type r) {
            return get_rank(dynamic_indices[r]);
        }
        using type = extents<single_slice_helper<get_rank(
            Rs)>::template static_extent<get_rank(Rs), ET>()...>;

        constexpr static std::array<rank_type, type::rank_dynamic()>
            dynamic_indices =
                zipper::detail::extents::dynamic_extents_indices_v<type>;

        template <typename>
        struct _dynamic_helper;
        template <rank_type... Ds>
        struct _dynamic_helper<std::integer_sequence<rank_type, Ds...>> {
            constexpr static auto get_extents(const ET &base,
                                              const Slices &...slices) -> type {
                return type{
                    single_slice_helper<get_dynamic_rank(Ds)>::template extent<
                        get_dynamic_rank(Ds), ET>(
                        zipper::detail::pack_index<get_dynamic_rank(Ds)>(
                            slices...),
                        base)...

                };
            }
        };
        using dynamic_helper =
            _dynamic_helper<decltype(std::make_integer_sequence<
                                     rank_type, type::rank_dynamic()>{})>;
        constexpr static auto get_extents(const ET &base,
                                          const Slices &...slices) -> type {
            return dynamic_helper::get_extents(base, slices...);
        }
    };
    using maker =
        _extents_maker<decltype(std::make_integer_sequence<rank_type, rank>{})>;
    using extents_type = typename maker::type;

    constexpr static auto get_extents(const ET &base, const Slices &...slices)
        -> extents_type {
        return maker::get_extents(base, slices...);
    }
};
template <zipper::concepts::Extents ET, typename... Slices>
using slice_extent_helper = _slice_extent_helper<
    ET, decltype(std::make_integer_sequence<rank_type, ET::rank()>{}),
    Slices...>;

}  // namespace _detail_slice

}  // namespace unary

template <zipper::concepts::Expression ExprType,
          typename... Slices>
struct detail::ExpressionTraits<unary::Slice<ExprType, Slices...>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          ExprType, true> {
    using Base = detail::ExpressionTraits<ExprType>;

    template <rank_type R>
    using get_slice_t =
        std::decay_t<std::tuple_element_t<R, std::tuple<Slices...>>>;

    using value_type = Base::value_type;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;

    using extents_helper =
        unary::_detail_slice::slice_extent_helper<typename Base::extents_type,
                                           Slices...>;
    using extents_type = typename extents_helper::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    constexpr static std::array<rank_type, Base::extents_type::rank()>
        actionable_indices = extents_helper::get_actionable(
            std::make_index_sequence<Base::extents_type::rank()>{});
};

namespace unary {
template <zipper::concepts::Expression ExprType,
          typename... Slices>
class Slice : public UnaryExpressionBase<Slice<ExprType, Slices...>,
                                       ExprType> {
   public:
    using self_type = Slice<ExprType, Slices...>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    using Base = UnaryExpressionBase<self_type, ExprType>;
    using Base::extent;
    using Base::expression;
    using ExpressionType = std::decay_t<ExprType>;
    using expr_traits = zipper::expression::detail::ExpressionTraits<ExpressionType>;
    using expr_extents_type = expr_traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    using slice_storage_type =
        std::tuple<_detail_slice::slice_helper<std::decay_t<Slices>>...>;

    constexpr static std::array<rank_type, expr_extents_type::rank()>
        actionable_indices = traits::actionable_indices;

    Slice(const Slice &) = default;
    Slice(Slice &&) = default;

    Slice &operator=(zipper::concepts::Expression auto const &v) {
        assign(v);
        return *this;
    }
    Slice(ExprType &b, const Slices &...slices)
        : Base(b, traits::extents_helper::get_extents(b.extents(), slices...)),
          m_slices(_detail_slice::slice_helper<std::decay_t<Slices>>(slices)...) {}

    template <rank_type K, typename... Args>
    index_type get_index(Args &&...a) const {
        const auto &s = std::get<K>(m_slices);
        constexpr static index_type index = actionable_indices[K];
        if constexpr (index == std::dynamic_extent) {
            return s.get_index();
        } else {
            const auto &v = zipper::detail::pack_index<index>(a...);
            return s.get_index(v);
        }
    }

    template <typename... Args, rank_type... ranks>
    auto _coeff(std::integer_sequence<rank_type, ranks...>,
                Args &&...idxs) const -> value_type {
        return expression().coeff(get_index<ranks>(idxs...)...);
    }
    template <typename... Args, rank_type... ranks>
    auto _coeff_ref(std::integer_sequence<rank_type, ranks...>, Args &&...idxs)
        -> value_type &
        requires(traits::is_assignable())
    {
        return expression().coeff_ref(get_index<ranks>(idxs...)...);
    }
    template <typename... Args, rank_type... ranks>
    auto _const_coeff_ref(std::integer_sequence<rank_type, ranks...>,
                          Args &&...idxs) const -> const value_type &
        requires(traits::is_referrable())
    {
        return expression().const_coeff_ref(get_index<ranks>(idxs...)...);
    }

    template <typename... Args>
    value_type coeff(Args &&...idxs) const {
        return _coeff(
            std::make_integer_sequence<rank_type, expr_extents_type::rank()>{},
            std::forward<Args>(idxs)...);
    }
    template <typename... Args>
    value_type &coeff_ref(Args &&...idxs)
        requires(traits::is_assignable())
    {
        return _coeff_ref(
            std::make_integer_sequence<rank_type, expr_extents_type::rank()>{},
            std::forward<Args>(idxs)...);
    }
    template <typename... Args>
    const value_type &const_coeff_ref(Args &&...idxs) const
        requires(traits::is_referrable())
    {
        return _const_coeff_ref(
            std::make_integer_sequence<rank_type, expr_extents_type::rank()>{},
            std::forward<Args>(idxs)...);
    }

    template <zipper::concepts::Expression V>
    void assign(const V &v)
        requires(
            traits::is_assignable() &&
            extents_traits::template is_convertable_from<
                typename zipper::expression::detail::ExpressionTraits<V>::extents_type>())
    {
        expression::detail::AssignHelper<V, self_type>::assign(v, *this);
    }

   private:
    slice_storage_type m_slices;
};

template <zipper::concepts::Expression ExprType, typename... Slices>
Slice(const ExprType &expr, Slices &&...)
    -> Slice<const ExprType, std::decay_t<Slices>...>;

template <zipper::concepts::Expression ExprType, typename... Slices>
Slice(ExprType &expr, Slices &&...)
    -> Slice<ExprType, std::decay_t<Slices>...>;

}  // namespace unary
}  // namespace zipper::expression
#endif
