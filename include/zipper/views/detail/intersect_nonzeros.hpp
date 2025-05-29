#if !defined(ZIPPER_VIEWS_DETAIL_INTERSECT_NONZEROS_HPP)
#define ZIPPER_VIEWS_DETAIL_INTERSECT_NONZEROS_HPP
#include <spdlog/spdlog.h>

#include "zipper/types.hpp"
namespace zipper::views::detail {

template <typename A, typename B>
struct intersect_nonzeros {
   public:
    using a_iterator = typename A::const_iterator;
    using b_iterator = typename B::const_iterator;

    struct iterator_ {
       public:
        using value_type = index_type;
        using iterator_category = std::forward_iterator_tag;
        using difference_type = int64_t;

        iterator_() = default;
        iterator_(const intersect_nonzeros& base)
            : base(&base), a_it(base.a.begin()), b_it(base.b.begin()) {
            // make sure the iterators initialize in the right state
            if (*a_it != *b_it) {
                operator++();
            }
        }

        iterator_(const intersect_nonzeros& base, int)
            : base(&base), a_it(base.a.end()), b_it(base.b.end()) {}
        iterator_(const iterator_&) = default;
        iterator_(iterator_&&) = default;
        iterator_& operator=(const iterator_& o) = default;
        iterator_& operator=(iterator_&& o) = default;
        iterator_& operator++(int) {
            iterator_ r = *this;
            ++(*this);
            return r;
        }
        iterator_& operator++() {
            if (*a_it == *b_it) {
                std::advance(a_it, 1);
                std::advance(b_it, 1);
            }
            while (a_it != base->a.end() && b_it != base->b.end()) {
                auto cmp = *a_it <=> *b_it;
                if (std::is_eq(cmp)) {
                    break;
                } else if (std::is_lt(cmp)) {
                    ++a_it;
                } else {
                    ++b_it;
                }
            }
            if (a_it == base->a.end()) {
                b_it = base->b.end();
            }
            if (b_it == base->b.end()) {
                a_it = base->a.end();
            }
            return *this;
        }
        bool operator==(const iterator_& other) const = default;
        value_type operator*() const { return *a_it; }

       private:
        const intersect_nonzeros* base = nullptr;
        a_iterator a_it;
        b_iterator b_it;
    };
    using iterator = iterator_;
    using const_iterator = iterator_;
    intersect_nonzeros(const A& a, const B& b) : a(a), b(b) {}
    intersect_nonzeros(const intersect_nonzeros&) = default;
    intersect_nonzeros(intersect_nonzeros&&) = default;

    iterator begin() const { return iterator(*this); }
    iterator end() const { return iterator(*this, 0); }
    iterator cbegin() const { return iterator(*this); }
    iterator cend() const { return iterator(*this, 0); }

   private:
    const A& a;
    const B& b;
};

template <typename A, typename B>
intersect_nonzeros(const A& a, const B& b) -> intersect_nonzeros<A, B>;

}  // namespace zipper::views::detail
#endif
