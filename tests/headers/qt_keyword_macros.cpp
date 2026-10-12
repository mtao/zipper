// Simulate Qt's keyword macros being defined before zipper is included.
#define emit
#define signals public
#define slots
#include <zipper/Matrix.hpp>
#include <zipper/MatrixBase.hxx>

#ifndef emit
#error "zipper must restore the emit macro"
#endif

auto main() -> int {
    zipper::Matrix<double, 2, 2> a{{1.0, 2.0}, {3.0, 4.0}};
    auto b = (a * a).eval();
    return b(0, 0) == 7.0 ? 0 : 1;
}
