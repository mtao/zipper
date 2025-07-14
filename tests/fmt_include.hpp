#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
#pragma GCC diagnostic ignored "-Wswitch-enum"
#if defined(__clang__)
#else
#pragma GCC diagnostic ignored "-Wabi-tag"
#pragma GCC diagnostic ignored "-Wmultiple-inheritance"
#endif

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop
