#include <iostream>
#include <zipper/Array.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>

#include "zipper/expression/nullary/Random.hpp"
// Try running this in the build folder:
// > $( ninja -t commands  examples/static_fma | head -n 1 | sed
// 's|static_fma.cpp.o|static_fma.cpp.s|g') -S -mavx -mfma then obtaining the
// output file assembly with > cat examples/static_fma.p/static_fma.cpp.s

__attribute__((noinline)) auto DOOP(const auto& a, const auto& b,
                                    const auto& c) {
    return (a.as_array() * (b + c).as_array()).eval();
}
__attribute__((noinline)) auto f() {
    auto R = zipper::expression::nullary::uniform_random<float>(
        zipper::extents<20, 20>{}, 0, 1);
    zipper::Matrix A = R;
    zipper::Matrix B = R;
    zipper::Matrix C = R;

    return DOOP(A, B, C);
}
//__attribute__((noinline)) auto g() {
//    auto R = zipper::expression::nullary::uniform_random<float>(
//        zipper::extents<20, 20>{}, 0, 1);
//    zipper::Array A = R;
//    zipper::Array B = R;
//    zipper::Array C = R;
//
//    return DOOP(A, B, C);
//}
__attribute__((noinline)) auto h() {
    auto R = zipper::expression::nullary::uniform_random<float>(zipper::extents<20>{},
                                                             0, 1);
    zipper::Vector A = R;
    zipper::Vector B = R;
    zipper::Vector C = R;

    return DOOP(A, B, C);
}

__attribute__((noinline)) auto l() {
    auto R = zipper::expression::nullary::uniform_random<float>(
        zipper::extents<4, 4>{}, 0, 1);
    zipper::Matrix A = R;
    zipper::Matrix B = R;
    zipper::Matrix C = R;

    return DOOP(A, B, C);
}
__attribute__((noinline)) auto m() {
    auto R = zipper::expression::nullary::uniform_random<float>(zipper::extents<4>{},
                                                             0, 1);
    zipper::Vector A = R;
    zipper::Vector B = R;
    zipper::Vector C = R;

    return DOOP(A, B, C);
}

int main(int, char*[]) {
    std::cout << f().norm() << std::endl;
    // std::cout << g().norm() << std::endl;
    std::cout << h().norm() << std::endl;
    std::cout << l().norm() << std::endl;
    std::cout << m().norm() << std::endl;
}
