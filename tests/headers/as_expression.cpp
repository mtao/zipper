// as_* on raw expressions needs only the target base's header.
#include <zipper/Array.hpp>
#include <zipper/VectorBase.hpp>

auto main() -> int {
    zipper::Array<double, 2, 3> a;
    a.fill(1.0);
    // Rank 1: (2, 3) -> (2).
    const auto sums = zipper::as_array(a.over<1>().sum());
    const auto as_vector = zipper::as_vector(a.over<1>().sum());
    // Squeezed: (1, 3) -> (3).
    zipper::Array<double, 1, 3> row;
    row.fill(2.0);
    const auto squeezed = zipper::as_vector(row.expression());
    return sums(0) == 3.0 && as_vector(1) == 3.0 && squeezed.extent(0) == 3 && squeezed(2) == 2.0
               ? 0
               : 1;
}
