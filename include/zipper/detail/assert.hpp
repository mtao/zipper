#if !defined(ZIPPER_DETAIL_ASSERT_HPP)
#define ZIPPER_DETAIL_ASSERT_HPP

// ZIPPER_ASSERT: ODR-safe replacement for assert() in header-defined templates.
//
// Standard assert() depends on NDEBUG, which can differ between translation
// units, causing ODR violations in header-only template code. ZIPPER_ASSERT
// is controlled by ZIPPER_DEBUG instead, which should be set consistently
// across the entire build via the build system (e.g. -DZIPPER_DEBUG).
//
// Behavior:
//   ZIPPER_DEBUG defined     -> evaluates expression, aborts on failure
//   ZIPPER_DEBUG not defined -> no-op (zero overhead)
//
// To enable debug assertions, add -DZIPPER_DEBUG to your compiler flags.

#if defined(ZIPPER_DEBUG)

#include <cstdio>
#include <cstdlib>

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ZIPPER_ASSERT(expr)                                                    \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::fprintf(stderr, "ZIPPER_ASSERT failed: %s [%s:%d]\n", #expr,       \
                   __FILE__, __LINE__);                                        \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#else

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ZIPPER_ASSERT(expr) ((void)0)

#endif // ZIPPER_DEBUG

#endif // ZIPPER_DETAIL_ASSERT_HPP
