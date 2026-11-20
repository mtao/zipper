#pragma once

#include "../assignment_source.hpp"
#include "../concepts/Expression.hpp"
#include "../concepts/Zipper.hpp"
#include "../expression/concepts/capabilities.hpp"
#include "../utils/extents/assignable_extents.hpp"

#include <concepts>
#include <type_traits>
#include <utility>

namespace zipper::detail {

// Preserve native expression/wrapper overloads: adaptation is only considered
// for explicitly opted-in foreign types. The helper keeps the adapter alive
// while its expression is evaluated, avoiding wrapper move-assignment/rebinding.
template <typename T>
concept AdaptedAssignmentSource =
    !concepts::Zipper<std::remove_cvref_t<T>> &&
    !concepts::Expression<std::remove_cvref_t<T>> &&
    requires(const std::remove_reference_t<T> &source) {
        assignment_source_traits<std::remove_cvref_t<T>>::get(source);
        requires(!std::is_reference_v<decltype(
            assignment_source_traits<std::remove_cvref_t<T>>::get(source))>);
        requires(concepts::Zipper<std::remove_cvref_t<decltype(
                     assignment_source_traits<std::remove_cvref_t<T>>::get(source))>> ||
                 concepts::Expression<std::remove_cvref_t<decltype(
                     assignment_source_traits<std::remove_cvref_t<T>>::get(source))>>);
    };

template <typename Adapter>
auto adapted_expression(Adapter &adapter) -> decltype(auto);

template <AdaptedAssignmentSource Source>
using assignment_adapter_t = std::remove_cvref_t<decltype(
    assignment_source_traits<std::remove_cvref_t<Source>>::get(
        std::declval<const std::remove_reference_t<Source> &>()))>;

template <AdaptedAssignmentSource Source>
using assignment_expression_t = std::remove_cvref_t<decltype(
    adapted_expression(std::declval<assignment_adapter_t<Source> &>()))>;

template <typename Source, typename Destination>
concept AdaptedAssignmentTo = AdaptedAssignmentSource<Source> &&
    concepts::Expression<std::remove_cvref_t<Destination>> &&
    (assignment_expression_t<Source>::extents_type::rank() ==
     std::remove_cvref_t<Destination>::extents_type::rank()) &&
    utils::extents::assignable_extents_v<
        typename assignment_expression_t<Source>::extents_type,
        typename std::remove_cvref_t<Destination>::extents_type> &&
    std::convertible_to<typename assignment_expression_t<Source>::value_type,
                        typename std::remove_cvref_t<Destination>::value_type> &&
    requires(Destination &destination, const assignment_expression_t<Source> &source) {
        destination.assign(source);
    };

template <typename Source, typename Destination>
concept AdaptedAssignmentFor = AdaptedAssignmentTo<Source, Destination> &&
    expression::concepts::WritableExpression<std::remove_cvref_t<Destination>>;

template <typename Adapter>
auto adapted_expression(Adapter &adapter) -> decltype(auto) {
    if constexpr (concepts::Zipper<std::remove_cvref_t<Adapter>>) {
        return adapter.expression();
    } else {
        return (adapter);
    }
}

template <typename Destination, typename Source>
    requires AdaptedAssignmentTo<Source, Destination>
void assign_adapted(Destination &destination, const Source &source) {
    auto adapter = assignment_source_traits<std::remove_cvref_t<Source>>::get(source);
    destination.assign(adapted_expression(adapter));
}

} // namespace zipper::detail
