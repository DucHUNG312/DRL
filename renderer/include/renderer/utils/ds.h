#pragma once

#include "renderer/core.h"
#include "renderer/cuda/core.h"

namespace lab
{

namespace renderer
{

namespace span_internal 
{

template <typename C>
LAB_CPU_GPU inline constexpr auto get_data_impl(C &c, char) noexcept -> decltype(c.data()) 
{
    return c.data();
}

template <typename C>
LAB_CPU_GPU inline constexpr auto get_data(C &c) noexcept -> decltype(get_data_impl(c, 0)) 
{
    return get_data_impl(c, 0);
}

template <typename C>
using has_size =
    std::is_integral<typename std::decay_t<decltype(std::declval<C &>().size())>>;

template <typename T, typename C>
using has_data =
    std::is_convertible<typename std::decay_t<decltype(get_data(std::declval<C &>()))> *,
                        T *const *>;

}

inline constexpr int64_t dynamic_extent = -1;

template <typename T>
class span 
{
public:
    template <typename C>
    using EnableIfConvertibleFrom =
        typename std::enable_if_t<span_internal::has_data<T, C>::value &&
                                  span_internal::has_size<C>::value>;

    template <typename U>
    using EnableIfConstView = typename std::enable_if_t<std::is_const_v<T>, U>;

    template <typename U>
    using EnableIfMutableView = typename std::enable_if_t<!std::is_const_v<T>, U>;

    using value_type = typename std::remove_cv_t<T>;
    using iterator = T *;
    using const_iterator = const T *;

    LAB_CPU_GPU
    span() : ptr(nullptr), n(0) {}
    LAB_CPU_GPU
    span(T *ptr, int64_t n) : ptr(ptr), n(n) {}
    template <int64_t N>
    LAB_CPU_GPU span(T (&a)[N]) : span(a, N) {}
    LAB_CPU_GPU
    span(std::initializer_list<value_type> v) : span(v.begin(), v.size()) {}

    // Explicit reference constructor for a mutable `span<T>` type. Can be
    // replaced with make_span() to infer the type parameter.
    template <typename V, typename X = EnableIfConvertibleFrom<V>,
              typename Y = EnableIfMutableView<V>>
    LAB_CPU_GPU explicit span(V &v) noexcept : span(v.data(), v.size()) {}

    // Hack: explicit constructors for std::vector to work around warnings
    // about calling a host function (e.g. vector::size()) form a
    // host+device function (the regular span constructor.)
    template <typename V>
    span(std::vector<V> &v) noexcept : span(v.data(), v.size()) {}
    template <typename V>
    span(const std::vector<V> &v) noexcept : span(v.data(), v.size()) {}

    // Implicit reference constructor for a read-only `span<const T>` type
    template <typename V, typename X = EnableIfConvertibleFrom<V>,
              typename Y = EnableIfConstView<V>>
    LAB_CPU_GPU constexpr span(const V &v) noexcept : span(v.data(), v.size()) {}

    LAB_CPU_GPU
    iterator begin() { return ptr; }
    LAB_CPU_GPU
    iterator end() { return ptr + n; }
    LAB_CPU_GPU
    const_iterator begin() const { return ptr; }
    LAB_CPU_GPU
    const_iterator end() const { return ptr + n; }

    LAB_CPU_GPU
    T &operator[](int64_t i) 
    {
        LAB_CHECK_LT(i, size());
        return ptr[i];
    }
    LAB_CPU_GPU
    const T &operator[](int64_t i) const 
    {
        LAB_CHECK_LT(i, size());
        return ptr[i];
    }

    LAB_CPU_GPU
    int64_t size() const { return n; };
    LAB_CPU_GPU
    bool empty() const { return size() == 0; }
    LAB_CPU_GPU
    T *data() { return ptr; }
    LAB_CPU_GPU
    const T *data() const { return ptr; }

    LAB_CPU_GPU
    T front() const { return ptr[0]; }
    LAB_CPU_GPU
    T back() const { return ptr[n - 1]; }

    LAB_CPU_GPU
    void remove_prefix(int64_t count) 
    {
        // assert(size() >= count);
        ptr += count;
        n -= count;
    }
    LAB_CPU_GPU
    void remove_suffix(int64_t count) 
    {
        // assert(size() > = count);
        n -= count;
    }

    LAB_CPU_GPU
    span subspan(int64_t pos, int64_t count = dynamic_extent) 
    {
        int64_t np = count < (size() - pos) ? count : (size() - pos);
        return span(ptr + pos, np);
    }

private:
    T *ptr;
    int64_t n;
};

template <int &...ExplicitArgumentBarrier, typename T>
LAB_CPU_GPU inline constexpr span<T> make_span(T* ptr, int64_t size) noexcept 
{
    return span<T>(ptr, size);
}

template <int &...ExplicitArgumentBarrier, typename T>
LAB_CPU_GPU inline span<T> make_span(T* begin, T* end) noexcept 
{
    return span<T>(begin, end - begin);
}

template <int &...ExplicitArgumentBarrier, typename T>
inline span<T> make_span(std::vector<T> &v) noexcept 
{
    return span<T>(v.data(), v.size());
}

template <int &...ExplicitArgumentBarrier, typename C>
LAB_CPU_GPU inline constexpr auto make_span(C &c) noexcept
    -> decltype(make_span(span_internal::get_data(c), c.size())) 
    {
    return make_span(span_internal::get_data(c), c.size());
}

template <int &...ExplicitArgumentBarrier, typename T, int64_t N>
LAB_CPU_GPU inline constexpr span<T> make_span(T (&array)[N]) noexcept {
    return span<T>(array, N);
}

template <int &...ExplicitArgumentBarrier, typename T>
LAB_CPU_GPU inline constexpr span<const T> make_const_span(T *ptr, int64_t size) noexcept 
{
    return span<const T>(ptr, size);
}

template <int &...ExplicitArgumentBarrier, typename T>
LAB_CPU_GPU inline span<const T> make_const_span(T *begin, T *end) noexcept 
{
    return span<const T>(begin, end - begin);
}

template <int &...ExplicitArgumentBarrier, typename T>
inline span<const T> make_const_span(const std::vector<T> &v) noexcept 
{
    return span<const T>(v.data(), v.size());
}

template <int &...ExplicitArgumentBarrier, typename C>
LAB_CPU_GPU inline constexpr auto make_const_span(const C &c) noexcept
    -> decltype(make_span(c)) {
    return make_span(c);
}

template <int &...ExplicitArgumentBarrier, typename T, int64_t N>
LAB_CPU_GPU inline constexpr span<const T> make_const_span(const T (&array)[N]) noexcept 
{
    return span<const T>(array, N);
}

}

}