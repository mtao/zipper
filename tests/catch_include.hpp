#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
#pragma GCC diagnostic ignored "-Wstrict-overflow"
#pragma GCC diagnostic ignored "-Weffc++"


#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wunsafe-buffer-usage"
#else
#pragma GCC diagnostic ignored "-Wabi-tag"
#pragma GCC diagnostic ignored "-Wmultiple-inheritance"
#endif
#include <catch2/catch_all.hpp>
#pragma GCC diagnostic pop
