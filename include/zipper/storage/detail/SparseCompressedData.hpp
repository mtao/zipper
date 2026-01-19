#if !defined(ZIPPER_STORAGE_DETAIL_SPARSECOMPRESSEDDATA_HPP)
#define ZIPPER_STORAGE_DETAIL_SPARSECOMPRESSEDDATA_HPP

#include <vector>
#include <zipper/concepts/Index.hpp>
#include <zipper/types.hpp>
namespace zipper::storage::detail {

template <typename T, rank_type N>
  requires(N >= 0)
struct SparseCompressedData : public SparseCompressedData<T, N - 1> {
  using Parent = SparseCompressedData<T, N - 1>;

  SparseCompressedData() = default;
  SparseCompressedData(SparseCompressedData &&) = default;
  auto operator=(SparseCompressedData &&) -> SparseCompressedData & = default;
  SparseCompressedData(const SparseCompressedData &) = default;
  auto operator=(const SparseCompressedData &)
      -> SparseCompressedData & = default;
  ~SparseCompressedData();
  using value_type = T;
  std::vector<std::tuple<index_type, size_t, size_t>> m_spans = {};

  template <zipper::concepts::IndexPack... Leftover>
    requires(sizeof...(Leftover) == N)
  auto __attribute__((pure)) coeff_(size_t start, size_t size, index_type index,
                                    Leftover... leftover) const -> value_type {
    auto spans = std::span(m_spans).subspan(start, size);

    auto it = std::lower_bound(
        spans.begin(), spans.end(), index,
        [](const std::tuple<index_type, size_t, size_t> &a,
           index_type b) -> auto { return std::get<0>(a) < b; });
    if (it != spans.end() && std::get<0>(*it) == index) {
      auto [i, mystart, mysize] = *it;
      return Parent::coeff_(mystart, mysize, leftover...);
    } else {
      return value_type(0);
    }
  }
  [[nodiscard]] auto size() const -> size_t { return m_spans.size(); }
  template <typename... Leftover>
    requires((sizeof...(Leftover) == N) &&
             zipper::concepts::IndexPack<Leftover...>)
  auto __attribute__((pure)) coeff(index_type index, Leftover... leftover) const
      -> value_type {
    return coeff_(0, m_spans.size(), index, leftover...);
  }

  template <zipper::concepts::IndexPack... Leftover>
    requires(sizeof...(Leftover) == N)
  auto insert_back(index_type index, Leftover... leftover) -> value_type & {
    return insert_back_(0, m_spans.size(), index, leftover...);
  }
  template <zipper::concepts::IndexPack... Leftover>
    requires(sizeof...(Leftover) == N)
  auto insert_back_(size_t /*start*/, size_t /*size*/, index_type index,
                    Leftover... leftover) -> value_type & {
    // assert(start + size == this->size() + 1);
    if (m_spans.empty()) {
      m_spans.emplace_back(index, Parent::size(), 1);
    } else {
      auto spans = std::span(m_spans); //.subspan(start, size);
      auto &[myindex, mystart, mysize] = spans.back();
      if (myindex == index) {
        mysize++;
      } else if (myindex < index) {
        m_spans.emplace_back(index, Parent::size(), 1);
      } else {
        throw std::invalid_argument("Inserted index out of order");
      }
    }
    auto &[i, mystart, mysize] = m_spans.back();
    return Parent::insert_back_(mystart, mysize, leftover...);
  }
};

template <typename T> struct SparseCompressedData<T, 0> {
  using value_type = T;
  std::vector<std::pair<index_type, T>> m_data = {};

  SparseCompressedData() = default;
  SparseCompressedData(SparseCompressedData &&) = default;
  auto operator=(SparseCompressedData &&) -> SparseCompressedData & = default;
  SparseCompressedData(const SparseCompressedData &) = default;
  auto operator=(const SparseCompressedData &)
      -> SparseCompressedData & = default;
  ~SparseCompressedData();

  auto __attribute__((pure)) coeff_(size_t start, size_t size,
                                    index_type index) const -> value_type {
    auto sp = std::span(m_data).subspan(start, size);
    auto it =
        std::lower_bound(sp.begin(), sp.end(), index,
                         [](const std::pair<index_type, T> &a,
                            index_type b) -> auto { return a.first < b; });
    if (it != sp.end() && it->first == index) {
      return it->second;
    } else {
      return value_type(0);
    }
  }
  [[nodiscard]] auto size() const -> size_t { return m_data.size(); }
  auto __attribute__((pure)) coeff(index_type index) const -> value_type {
    return coeff_(0, m_data.size(), index);
  }
  auto insert_back(index_type index) -> value_type & {
    return insert_back_(0, m_data.size(), index);
  }
  auto insert_back_(size_t /*start*/, size_t /*size*/, index_type index)
      -> value_type & {
    // assert(start + size == this->size() + 1);
    if (m_data.empty()) {
      return std::get<1>(m_data.emplace_back(index, 0));
    } else {
      // auto sp = std::span(m_data);  //.subspan(start, size);
      //  if (std::get<0>(sp.back()) < index) {
      return std::get<1>(m_data.emplace_back(index, 0));
      //} else {
      //    throw std::invalid_argument(
      //        "Tried inserting an index into the back twice");
      //}
    }
  }
};

template <typename T, rank_type N>
  requires(N >= 0)
SparseCompressedData<T, N>::~SparseCompressedData() = default;
template <typename T>
SparseCompressedData<T, 0>::~SparseCompressedData() = default;
} // namespace zipper::storage::detail

#endif
