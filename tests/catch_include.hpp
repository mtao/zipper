#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
#pragma GCC diagnostic ignored "-Weffc++"

// float equality is generally ok, but in this library the unit tests are mostly for flop exact computation
#pragma GCC diagnostic ignored "-Wfloat-equal"

#if defined(__clang__)
#else
#pragma GCC diagnostic ignored "-Wabi-tag"
#pragma GCC diagnostic ignored "-Wmultiple-inheritance"
#endif
#include <catch2/catch_all.hpp>
#pragma GCC diagnostic pop
