#if !defined(UVL_TENSOR_HPP)
#define UVL_TENSOR_HPP

#include "TensorBase.hpp"
#include "concepts/TensorBaseDerived.hpp"
#include "storage/PlainObjectStorage.hpp"
#include "uvl/types.hpp"
namespace uvl {

template <typename ValueType, index_type... Dims>
    requires(sizeof...(Dims) > 0)
class Tensor
    : public TensorBase<
          storage::PlainObjectStorage<ValueType, uvl::extents<Dims...>>> {
   public:
    using Base = TensorBase<
        storage::PlainObjectStorage<ValueType, uvl::extents<Dims...>>>;
    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using Base::extent;
    using Base::extents;

    template <concepts::ViewDerived Other>
    Tensor(const Other& other) : Base(other) {}
    template <concepts::TensorBaseDerived Other>
    Tensor(const Other& other) : Base(other) {}
    template <typename... Args>
    Tensor(Args&&... args)
        requires((std::is_convertible_v<Args, index_type> && ...))
        : Base(uvl::extents<Dims...>(std::forward<Args>(args)...)) {}
    template <index_type... indices>
    Tensor(const uvl::extents<indices...>& e) : Base(e) {}
    using Base::operator=;

    template <typename... Args>
    const value_type& operator()(Args&&... idxs) const

    {
        return view()(std::forward<Args>(idxs)...);
    }
    template <typename... Args>
    value_type& operator()(Args&&... idxs)

    {
        return view()(std::forward<Args>(idxs)...);
    }
};
}  // namespace uvl

namespace uvl::views {

template <typename ValueType, index_type... Dims>
struct detail::ViewTraits<Tensor<ValueType, Dims...>>
    : public detail::ViewTraits<
          uvl::storage::PlainObjectStorage<ValueType, uvl::extents<Dims...>>> {
};
}  // namespace uvl::views
#endif
