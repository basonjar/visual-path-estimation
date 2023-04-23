#pragma once

#include "pch.h"

namespace detail {

    template<typename T>
    class EnumerateIterator {
    private:
        size_t index{};
        T it;

    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = std::pair<size_t, typename T::value_type>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        EnumerateIterator() = default;
        EnumerateIterator(size_t index, T it) : index{index}, it{std::move(it)} {}

        value_type operator*() const {
            return std::make_pair(index, *it);
        }

        auto& operator++() {
            ++index;
            ++it;
            return *this;
        }

        auto operator++(int) {
            auto copy = *this;
            ++*this;
            return copy;
        }

        auto operator==(EnumerateIterator const& other) const { return index == other.index; }
        auto operator!=(EnumerateIterator const& other) const { return !(*this == other); }
    };

    template<typename C>
    struct collect_helper {};

    template<typename C, std::ranges::range R>
    auto operator|(R&& r, collect_helper<C>) {
        return C{std::ranges::begin(r), std::ranges::end(r)};
    }

    struct enumerate_helper {};

    template<std::ranges::range R>
        requires std::ranges::sized_range<R>
    auto operator|(R&& r, enumerate_helper) {
        auto begin = EnumerateIterator{0uz, std::ranges::begin(r)};
        auto end = EnumerateIterator{std::ranges::size(r), std::ranges::end(r)};
        return std::ranges::subrange{begin, end};
    }

}// namespace detail

template<std::ranges::range C>
auto collect() {
    return detail::collect_helper<C>{};
}

auto enumerate() {
    return detail::enumerate_helper{};
}
