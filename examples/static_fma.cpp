#include <iostream>
#include <uvl/Array.hpp>
#include <uvl/Matrix.hpp>
#include <uvl/Vector.hpp>

#include "uvl/views/nullary/RandomView.hpp"
// Try running this in the build folder:
// > $( ninja -t commands  examples/static_fma | head -n 1 | sed 's|static_fma.cpp.o|static_fma.cpp.s|g') -S -mavx -mfma
// then obtaining the output file assembly with 
// > cat examples/static_fma.p/static_fma.cpp.s

__attribute__((noinline)) auto DOOP(const auto& a, const auto& b,
                                    const auto& c) {
    return (a.as_array() * (b + c).as_array()).eval();
}
__attribute__((noinline)) auto f() {
    auto R = uvl::views::nullary::uniform_random_view<float>(
        uvl::extents<20, 20>{}, 0, 1);
    uvl::Matrix A = R;
    uvl::Matrix B = R;
    uvl::Matrix C = R;

    return DOOP(A, B, C);
}
//__attribute__((noinline)) auto g() {
//    auto R = uvl::views::nullary::uniform_random_view<float>(
//        uvl::extents<20, 20>{}, 0, 1);
//    uvl::Array A = R;
//    uvl::Array B = R;
//    uvl::Array C = R;
//
//    return DOOP(A, B, C);
//}
__attribute__((noinline)) auto h() {
    auto R = uvl::views::nullary::uniform_random_view<float>(uvl::extents<20>{},
                                                             0, 1);
    uvl::Vector A = R;
    uvl::Vector B = R;
    uvl::Vector C = R;

    return DOOP(A, B, C);
}

__attribute__((noinline)) auto l() {
    auto R = uvl::views::nullary::uniform_random_view<float>(
        uvl::extents<4, 4>{}, 0, 1);
    uvl::Matrix A = R;
    uvl::Matrix B = R;
    uvl::Matrix C = R;

    return DOOP(A, B, C);
}
__attribute__((noinline)) auto m() {
    auto R = uvl::views::nullary::uniform_random_view<float>(uvl::extents<4>{},
                                                             0, 1);
    uvl::Vector A = R;
    uvl::Vector B = R;
    uvl::Vector C = R;

    return DOOP(A, B, C);
}

int main(int argc, char* argv[]) {
    std::cout << f().norm() << std::endl;
    // std::cout << g().norm() << std::endl;
    std::cout << h().norm() << std::endl;
    std::cout << l().norm() << std::endl;
    std::cout << m().norm() << std::endl;
}
