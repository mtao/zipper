#if !defined(ZIPPER_STORAGE_DETAIL_SPARSE_COMPRESSED_DATA_HPP)
#define ZIPPER_STORAGE_DETAIL_SPARSE_COMPRESSED_DATA_HPP

#include <cstdint>
#include <vector>
#include <zipper/concepts/IndexPackLike.hpp>
#include <zipper/types.hpp>
namespace zipper::storage::detail {

template <typename T, rank_type N>
    requires(N >= 0)
struct SparseCompressedData : public SparseCompressedData<T, N - 1> {
    using Parent = SparseCompressedData<T, N - 1>;

    using value_type = T;
    std::vector<std::tuple<index_type, size_t, size_t>> m_spans;

    template <typename... Leftover>
        requires((sizeof...(Leftover) == N) &&
                 zipper::concepts::IndexPackLike<Leftover...>)
    value_type coeff_(size_t start, size_t size, index_type index,
                      Leftover... leftover) const {
        auto spans = std::span(m_spans).subspan(start, size);

        auto it =
            std::lower_bound(spans.begin(), spans.end(), index,
                             [](const std::tuple<index_type, size_t, size_t>& a,
                                index_type b) { return std::get<0>(a) < b; });
        if (it != spans.end() && std::get<0>(*it) == index) {
            auto [i, mystart, mysize] = *it;
            return Parent::coeff_(mystart, mysize, leftover...);
        } else {
            return value_type(0);
        }
    }
    size_t size() const { return m_spans.size(); }
    template <typename... Leftover>
        requires((sizeof...(Leftover) == N) &&
                 zipper::concepts::IndexPackLike<Leftover...>)
    value_type coeff(index_type index, Leftover... leftover) const {
        return coeff_(0, m_spans.size(), index, leftover...);
    }

    template <typename... Leftover>
        requires((sizeof...(Leftover) == N) &&
                 zipper::concepts::IndexPackLike<Leftover...>)
    value_type& insert_back(index_type index, Leftover... leftover) {
        return insert_back_(0, m_spans.size(), index, leftover...);
    }
    template <typename... Leftover>
        requires((sizeof...(Leftover) == N) &&
                 zipper::concepts::IndexPackLike<Leftover...>)

    value_type& insert_back_(size_t start, size_t size, index_type index,
                             Leftover... leftover) {
        // assert(start + size == this->size() + 1);
        if (m_spans.empty()) {
            m_spans.emplace_back(index, Parent::size(), 1);
        } else {
            auto spans = std::span(m_spans);  //.subspan(start, size);
            auto& [myindex, mystart, mysize] = spans.back();
            if (myindex == index) {
                mysize++;
            } else if (myindex < index) {
                m_spans.emplace_back(index, Parent::size(), 1);
            } else {
                throw std::invalid_argument("Inserted index out of order");
            }
        }
        auto& [i, mystart, mysize] = m_spans.back();
        return Parent::insert_back_(mystart, mysize, leftover...);
    }
};

template <typename T>
struct SparseCompressedData<T, 0> {
    using value_type = T;
    std::vector<std::pair<index_type, T>> m_data;

    value_type coeff_(size_t start, size_t size, index_type index) const {
        auto sp = std::span(m_data).subspan(start, size);
        auto it = std::lower_bound(sp.begin(), sp.end(), index,
                                   [](const std::pair<index_type, T>& a,
                                      index_type b) { return a.first < b; });
        if (it != sp.end() && it->first == index) {
            return it->second;
        } else {
            return value_type(0);
        }
    }
    size_t size() const { return m_data.size(); }
    value_type coeff(index_type index) const {
        return coeff_(0, m_data.size(), index);
    }
    value_type& insert_back(index_type index) {
        return insert_back_(0, m_data.size(), index);
    }
    value_type& insert_back_(size_t start, size_t size, index_type index) {
        // assert(start + size == this->size() + 1);
        if (m_data.empty()) {
            return std::get<1>(m_data.emplace_back(index, 0));
        } else {
            auto sp = std::span(m_data);  //.subspan(start, size);
            // if (std::get<0>(sp.back()) < index) {
            return std::get<1>(m_data.emplace_back(index, 0));
            //} else {
            //    throw std::invalid_argument(
            //        "Tried inserting an index into the back twice");
            //}
        }
    }
};
}  // namespace zipper::storage::detail

#endif
