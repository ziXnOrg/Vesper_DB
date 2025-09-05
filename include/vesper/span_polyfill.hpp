#pragma once

// Minimal span polyfill for GCC 11 compatibility
// GCC 11 has a broken std::span implementation, so we provide a minimal one

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

#if defined(__GNUC__) && __GNUC__ < 12
namespace std {
    template<typename T, std::size_t Extent = std::size_t(-1)>
    class span {
    public:
        using element_type = T;
        using value_type = std::remove_cv_t<T>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;
        using iterator = pointer;
        using const_iterator = const_pointer;

        constexpr span() noexcept : data_(nullptr), size_(0) {}
        constexpr span(pointer ptr, size_type count) : data_(ptr), size_(count) {}
        constexpr span(pointer first, pointer last) : data_(first), size_(last - first) {}
        
        template<std::size_t N>
        constexpr span(T (&arr)[N]) noexcept : data_(arr), size_(N) {}
        
        // Constructor from std::vector
        template<typename Container,
                 typename = std::enable_if_t<
                     std::is_same_v<std::remove_const_t<T>, 
                                   typename Container::value_type>>>
        constexpr span(Container& c) noexcept : data_(c.data()), size_(c.size()) {}
        
        template<typename Container,
                 typename = std::enable_if_t<
                     std::is_same_v<std::remove_const_t<T>, 
                                   typename Container::value_type>>>
        constexpr span(const Container& c) noexcept : data_(c.data()), size_(c.size()) {}

        constexpr pointer data() const noexcept { return data_; }
        constexpr size_type size() const noexcept { return size_; }
        constexpr size_type size_bytes() const noexcept { return size_ * sizeof(T); }
        constexpr bool empty() const noexcept { return size_ == 0; }

        constexpr reference operator[](size_type idx) const { return data_[idx]; }
        constexpr reference front() const { return data_[0]; }
        constexpr reference back() const { return data_[size_ - 1]; }

        constexpr iterator begin() const noexcept { return data_; }
        constexpr iterator end() const noexcept { return data_ + size_; }

        constexpr span first(size_type count) const { return {data_, count}; }
        constexpr span last(size_type count) const { return {data_ + size_ - count, count}; }
        constexpr span subspan(size_type offset, size_type count = size_type(-1)) const {
            return {data_ + offset, count == size_type(-1) ? size_ - offset : count};
        }

    private:
        pointer data_;
        size_type size_;
    };

    // Deduction guides
    template<typename T, std::size_t N>
    span(T (&)[N]) -> span<T, N>;
    
    template<typename T>
    span(T*, std::size_t) -> span<T>;
}
#else
#include <span>
#endif