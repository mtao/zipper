#include "zipper/detail/ExtentsTraits.hpp"
#if !defined(ZIPPER_VECTOR_HPP)
#define ZIPPER_VECTOR_HPP

#include "VectorBase.hpp"
#include "storage/PlainObjectStorage.hpp"
#include "zipper/types.hpp"
namespace zipper {

template <typename ValueType, index_type Rows>
class Vector
    : public VectorBase<
          storage::PlainObjectStorage<ValueType, zipper::extents<Rows>>> {
   public:
    using Base = VectorBase<
        storage::PlainObjectStorage<ValueType, zipper::extents<Rows>>>;
    using Base::view;
    using view_type = Base::view_type;
    using value_type = Base::value_type;
    using extents_type = Base::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;
    using Base::extent;
    using Base::extents;

    using span_type =
        VectorBase<storage::SpanStorage<ValueType, zipper::extents<Rows>>>;
    using const_span_type =
        VectorBase<storage::SpanStorage<const ValueType, zipper::extents<Rows>>>;

    Vector() = default;
    Vector(const Vector& o) = default;
    Vector& operator=(const Vector& o) = default;
    Vector(Vector&& o) = default;
    // Vector(Vector&& o):Base(std::move(o.view())) {
    // }
    Vector& operator=(Vector&& o) {
        view().operator=(std::move(o.view()));
        return *this;
    }
    Vector(index_type size)
        requires(extents_traits::is_dynamic)
        : Base(zipper::extents<Rows>(size)) {}

#if defined(NDEBUG)
    Vector(index_type)
#else
    Vector(index_type rows)
#endif
        requires(extents_traits::is_static)
        : Base() {
        assert(rows == extent(0));
    }


    template <index_type R2>
    Vector(const Vector<value_type, R2>& other) : Base(other.view()) {}

    template <concepts::VectorBaseDerived Other>
    Vector(const Other& other) : Base(other) {}

    template <concepts::ViewDerived Other>
    Vector(const Other& other) : Base(other) {}

    template <typename T>
    Vector(const std::initializer_list<T>& l)
        requires(extents_traits::is_static)
    {
        assert(l.size() == extent(0));
        std::ranges::copy(l, begin());
    }
    template <typename T>
    Vector(const std::initializer_list<T>& l)
        requires(extents_traits::is_dynamic)
        : Base(extents_type(l.size())) {
        std::ranges::copy(l, begin());
    }
    template <typename T>
    Vector& operator=(const std::initializer_list<T>& l)
        requires(extents_traits::is_static)
    {
        assert(l.size() == extent(0));
        std::ranges::copy(l, begin());
        return *this;
    }
    template <typename T>
    Vector& operator=(const std::initializer_list<T>& l)
        requires(extents_traits::is_dynamic)
    {
        view().resize(extents_type(l.size()));
        std::ranges::copy(l, begin());
        return *this;
    }

    auto begin() { return view().begin(); }
    auto end() { return view().end(); }
    auto begin() const { return view().begin(); }
    auto end() const { return view().end(); }

    using Base::operator=;
};

template <concepts::VectorViewDerived MB>
Vector(const MB& o)
    -> Vector<std::decay_t<typename MB::value_type>, MB::extents_type::static_extent(0)>;
template <concepts::VectorBaseDerived MB>
Vector(const MB& o)
    -> Vector<std::decay_t<typename MB::value_type>, MB::extents_type::static_extent(0)>;
}  // namespace zipper

namespace zipper::views {

template <typename ValueType, index_type Rows>
struct detail::ViewTraits<Vector<ValueType, Rows>>
    : public detail::ViewTraits<
          zipper::storage::PlainObjectStorage<ValueType, extents<Rows>>> {};
}  // namespace zipper::views
#endif
