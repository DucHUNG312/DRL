#pragma once

#include "lab/core.h"

namespace lab
{
namespace utils
{
/**
 * @brief Wrapper of std::vector
 */
template<typename T>
class Shape 
{
public:
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;
    using size_type = size_t;
    using value_type = T;
    using reverse_iterator = std::reverse_iterator<iterator>;

    LAB_ARG(std::vector<T>, data);
public:
    constexpr Shape() : data_({0}) {}

    constexpr Shape(const T& ele) : data_({ele}) {}

    template<typename A>
    constexpr Shape(const std::vector<T, A>& vec) 
        : data_(vec)  
    {
        static_assert(!std::is_same<T, bool>::value, "Shape<bool> cannot be constructed from a std::vector<bool> bitfield.");
    }

    constexpr Shape(torch::ArrayRef<T> arr) 
        : data_(arr.begin(), arr.end()) {}

    constexpr Shape(const Shape& other) 
        : data_(other.data_) {}

    constexpr Shape(Shape&& other) noexcept 
        : data_(std::move(other.data_)) {}

    constexpr Shape(const T* data, size_t length) 
        : data_(data, data + length) {}

    template <size_t N>
    constexpr Shape(const std::array<T, N>& arr)
        : data_(arr.data(), arr.data() + N) {}

    template <size_t N>
    constexpr Shape(const T (&arr)[N]) : data_(arr, arr + N) {}

    Shape<T>& operator=(const Shape<T>& other) 
    {
        if (this != &other)
            data_ = other.data_;
        return *this;
    }

    Shape<T>& operator=(Shape<T>&& other) noexcept 
    {
        if (this != &other)
            data_ = std::move(other.data_);
        return *this;
    }

    constexpr T& operator[](size_t idx) 
    {
        return data_[idx];
    }

    constexpr const T& operator[](size_t idx) const 
    {
        return data_[idx];
    }

    constexpr iterator begin() 
    {
        return data_.begin();
    }

    constexpr iterator end() 
    {
        return data_.end();
    }

    constexpr const_iterator cbegin() const 
    {
        return data_.cbegin();
    }

    constexpr const_iterator cend() const 
    {
        return data_.cend();
    }

    constexpr size_t size() const
    {
        return data_.size();
    }
    
    constexpr bool empty() const 
    {
        return size() == 0;
    }

    constexpr T& at(size_t idx) const 
    {
        if (idx >= data_.size())
            LAB_LOG_FATAL("Index out of range");
        return data_[idx];
    }

    constexpr const T& front() const 
    {
        LAB_CHECK(!empty());
        return data_[0];
    }

    constexpr const T& back() const 
    {
        LAB_CHECK(!empty());
        return data_[size() - 1];
    }

    constexpr bool equals(const Shape<T>& rhs) const 
    {
        return data_ == rhs.data();
    }

    constexpr bool equals(torch::ArrayRef<T> arr) const 
    {
        return data_ == arr;
    }

    constexpr Shape<T> slice(size_t n, size_t m) const 
    {
        LAB_CHECK_LE(n + m, size());
        return Shape<T>(data_.data() + n, m);
    }

    constexpr Shape<T> slice(size_t n) const 
    {
        LAB_CHECK_LE(n, size());
        return slice(n, size() - n);
    }

    constexpr torch::ArrayRef<T> to_torch() const 
    {
        return torch::ArrayRef<T>(data_);
    }

    friend std::ostream& operator<<(std::ostream& os, const Shape& arr) 
    {
        os << "[";
        for (size_t i = 0; i < arr.data_.size(); i++) 
        {
            if (i != 0) os << ", ";
            os << arr.data_[i];
        }
        os << "]";
        return os;
    }
};

template <typename T>
LAB_FORCE_INLINE Shape<T> make_shape(const T& ele) 
{
    return ele;
}

template <typename T>
LAB_FORCE_INLINE Shape<T> make_shape(const T* data, size_t length) 
{
    return Shape<T>(data, length);
}

template <typename T>
LAB_FORCE_INLINE Shape<T> make_shape(const std::vector<T>& vec) 
{
    return vec;
}

template <typename T, std::size_t N>
LAB_FORCE_INLINE Shape<T> make_shape(const std::array<T, N>& arr) 
{
    return arr;
}

template <typename T>
LAB_FORCE_INLINE Shape<T> make_shape(const Shape<T>& shape) 
{
    return shape;
}

template <typename T>
LAB_FORCE_INLINE Shape<T>& make_shape(Shape<T>& shape) 
{
    return shape;
}

template <typename T>
LAB_FORCE_INLINE Shape<T> make_shape(torch::ArrayRef<T> arr) 
{
    return arr;
}

template <typename T>
LAB_FORCE_INLINE bool operator==(const Shape<T>& a1, const Shape<T>& a2) 
{
    return a1.equals(a2);
}

template <typename T>
LAB_FORCE_INLINE bool operator!=(const Shape<T>& a1, const Shape<T>& a2) 
{
    return !a1.equals(a2);
}

template <typename T>
LAB_FORCE_INLINE bool operator==(const std::vector<T>& a1, const Shape<T>& a2) 
{
    return Shape<T>(a1).equals(a2);
}

template <typename T>
LAB_FORCE_INLINE bool operator!=(const std::vector<T>& a1, const Shape<T>& a2) 
{
    return !Shape<T>(a1).equals(a2);
}

template <typename T>
LAB_FORCE_INLINE bool operator==(const Shape<T>& a1, const std::vector<T>& a2) 
{
    return a1.equals(Shape<T>(a2));
}

template <typename T>
LAB_FORCE_INLINE bool operator!=(const Shape<T>& a1, const std::vector<T>& a2) 
{
    return !a1.equals(Shape<T>(a2));
}

template <typename T>
LAB_FORCE_INLINE bool operator==(const torch::ArrayRef<T>& a1, const Shape<T>& a2) 
{
    return Shape<T>(a1).equals(a2);
}

template <typename T>
LAB_FORCE_INLINE bool operator!=(const torch::ArrayRef<T>& a1, const Shape<T>& a2) 
{
    return !Shape<T>(a1).equals(a2);
}

template <typename T>
LAB_FORCE_INLINE bool operator==(const Shape<T>& a1, const torch::ArrayRef<T>& a2) 
{
    return a1.equals(Shape<T>(a2));
}

template <typename T>
LAB_FORCE_INLINE bool operator!=(const Shape<T>& a1, const torch::ArrayRef<T>& a2) 
{
    return !a1.equals(Shape<T>(a2));
}

using IShape = Shape<int64_t>;
using DShape = Shape<double>;
using FShape = Shape<float>;

}
}