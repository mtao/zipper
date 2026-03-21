#if !defined(ZIPPER_DETAIL_NO_UNIQUE_ADDRESS_HPP)
#define ZIPPER_DETAIL_NO_UNIQUE_ADDRESS_HPP

/// Portable `[[no_unique_address]]` attribute.
///
/// MSVC does not honour the standard `[[no_unique_address]]` attribute until
/// very recent versions; it requires `[[msvc::no_unique_address]]` instead.
/// This macro selects the correct spelling so that empty members (e.g.
/// statically-sized layout mappings, integral_constants) are truly zero-sized.
#if defined(_MSC_VER)
#define ZIPPER_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define ZIPPER_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

#endif
