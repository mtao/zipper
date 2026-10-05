#if !defined(ZIPPER_STORAGE_LAYOUT_PERMUTED_HPP)
#define ZIPPER_STORAGE_LAYOUT_PERMUTED_HPP

/// @file layout_permuted.hpp
/// @brief A layout mapping that permutes the dimensions of another mapping.
///
/// `layout_permuted<ChildMapping, Perm...>::mapping<E>` presents a child
/// mapping with its dimensions reordered: output dimension `d` is child
/// dimension `Perm[d]`. It is what a transpose (or any dimension permutation)
/// of a strided view is, keeping the permutation — and so the child's layout,
/// e.g. which dimension is contiguous — in the type, where a layout_stride
/// would only keep run-time strides.
///
/// Satisfies the mdspan LayoutMapping requirements, including
/// `submdspan_mapping` (a slice of a permuted mapping is a layout_stride).

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "zipper/storage/layout_types.hpp"
#include "zipper/types.hpp"

namespace zipper::storage {

template <typename ChildMapping, std::size_t... Perm>
struct layout_permuted {
    static_assert(sizeof...(Perm) == ChildMapping::extents_type::rank(),
                  "layout_permuted: one entry per child dimension");

    /// Output dimension d reads child dimension perm[d].
    static constexpr std::array<std::size_t, sizeof...(Perm)> perm{{Perm...}};

    template <typename Extents>
    class mapping {
      public:
        using extents_type = Extents;
        using index_type = typename extents_type::index_type;
        using size_type = typename extents_type::size_type;
        using rank_type = typename extents_type::rank_type;
        using layout_type = layout_permuted;
        using child_mapping_type = ChildMapping;

        static_assert(extents_type::rank() == sizeof...(Perm));

        constexpr mapping() = default;
        constexpr explicit mapping(const ChildMapping &child)
          : m_child(child),
            m_extents(child.extents().extent(Perm)...) {}

        constexpr auto extents() const -> const extents_type & {
            return m_extents;
        }
        constexpr auto child() const -> const ChildMapping & { return m_child; }

        template <typename... Idx>
            requires(sizeof...(Idx) == extents_type::rank())
        constexpr auto operator()(Idx... idx) const -> index_type {
            // child index c = output index d with perm[d] == c
            const std::array<index_type, sizeof...(Idx)> out{
                static_cast<index_type>(idx)...};
            return [&]<std::size_t... C>(std::index_sequence<C...>) {
                return static_cast<index_type>(m_child(out[inverse()[C]]...));
            }(std::make_index_sequence<sizeof...(Idx)>{});
        }

        constexpr auto required_span_size() const -> index_type {
            return static_cast<index_type>(m_child.required_span_size());
        }
        constexpr auto stride(rank_type r) const -> index_type {
            return static_cast<index_type>(m_child.stride(perm[r]));
        }

        static constexpr auto is_always_unique() -> bool {
            return ChildMapping::is_always_unique();
        }
        static constexpr auto is_always_exhaustive() -> bool {
            return ChildMapping::is_always_exhaustive();
        }
        static constexpr auto is_always_strided() -> bool {
            return ChildMapping::is_always_strided();
        }
        constexpr auto is_unique() const -> bool { return m_child.is_unique(); }
        constexpr auto is_exhaustive() const -> bool {
            return m_child.is_exhaustive();
        }
        constexpr auto is_strided() const -> bool { return m_child.is_strided(); }

        friend constexpr auto operator==(const mapping &a, const mapping &b)
            -> bool {
            return a.m_child == b.m_child;
        }

        /// Slicing goes through the equivalent layout_stride mapping.
        template <typename... Slices>
        friend constexpr auto submdspan_mapping(const mapping &m,
                                                Slices... slices) {
            using stride_mapping =
                typename layout_stride::template mapping<extents_type>;
            std::array<index_type, extents_type::rank()> strides{};
            for (rank_type r = 0; r < extents_type::rank(); ++r) {
                strides[r] = m.stride(r);
            }
            return submdspan_mapping(stride_mapping(m.extents(), strides),
                                     slices...);
        }

      private:
        /// inverse()[c] = the output dimension reading child dimension c.
        static constexpr auto inverse() {
            std::array<std::size_t, sizeof...(Perm)> inv{};
            for (std::size_t d = 0; d < sizeof...(Perm); ++d) {
                inv[perm[d]] = d;
            }
            return inv;
        }

        ChildMapping m_child{};
        extents_type m_extents{};
    };
};

} // namespace zipper::storage

#endif
