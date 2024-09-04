#pragma once

#include "renderer/core.h"
#include "renderer/math/math.h"

namespace lab
{

namespace math
{

namespace internal
{

template <typename T>
std::string to_string2(T x, T y);
template <typename T>
std::string to_string3(T x, T y, T z);

extern template std::string internal::to_string2(float, float);
extern template std::string internal::to_string2(double, double);
extern template std::string internal::to_string2(int, int);
extern template std::string internal::to_string3(float, float, float);
extern template std::string internal::to_string3(double, double, double);
extern template std::string internal::to_string3(int, int, int);

}

namespace 
{

// TupleLength Definition
template <typename T>
struct TupleLength 
{
    using type = float;
};

template <>
struct TupleLength<double> 
{
    using type = double;
};

template <>
struct TupleLength<long double> 
{
    using type = long double;
};

}  // anonymous namespace

template <template <typename> class Child, typename T>
class Tuple2 
{
public:
    static const int n_dimensions = 2;

    Tuple2() = default;
    LAB_CPU_GPU
    Tuple2(T x, T y) : x(x), y(y) { LAB_CHECK(!has_nan()); }
    LAB_CPU_GPU
    bool has_nan() const { return math::is_nan(x) || math::is_nan(y); }
#ifdef LAB_DEBUG
    // The default versions of these are fine for release builds; for debug
    // we define them so that we can add the Assert checks.
    LAB_CPU_GPU
    Tuple2(Child<T> c) 
    {
        LAB_CHECK(!c.has_nan());
        x = c.x;
        y = c.y;
    }
    LAB_CPU_GPU
    Child<T> &operator=(Child<T> c) 
    {
        LAB_CHECK(!c.has_nan());
        x = c.x;
        y = c.y;
        return static_cast<Child<T> &>(*this);
    }
#endif

    template <typename U>
    LAB_CPU_GPU auto operator+(Child<U> c) const -> Child<decltype(T{} + U{})> 
    {
        LAB_CHECK(!c.has_nan());
        return {x + c.x, y + c.y};
    }
    template <typename U>
    LAB_CPU_GPU Child<T> &operator+=(Child<U> c) 
    {
        LAB_CHECK(!c.has_nan());
        x += c.x;
        y += c.y;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    LAB_CPU_GPU auto operator-(Child<U> c) const -> Child<decltype(T{} - U{})> 
    {
        LAB_CHECK(!c.has_nan());
        return {x - c.x, y - c.y};
    }
    template <typename U>
    LAB_CPU_GPU Child<T> &operator-=(Child<U> c) 
    {
        LAB_CHECK(!c.has_nan());
        x -= c.x;
        y -= c.y;
        return static_cast<Child<T> &>(*this);
    }

    LAB_CPU_GPU
    bool operator==(Child<T> c) const { return x == c.x && y == c.y; }
    LAB_CPU_GPU
    bool operator!=(Child<T> c) const { return x != c.x || y != c.y; }

    template <typename U>
    LAB_CPU_GPU auto operator*(U s) const -> Child<decltype(T{} * U{})> 
    {
        return {s * x, s * y};
    }
    template <typename U>
    LAB_CPU_GPU Child<T> &operator*=(U s) 
    {
        LAB_CHECK(!math::is_nan(s));
        x *= s;
        y *= s;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    LAB_CPU_GPU auto operator/(U d) const -> Child<decltype(T{} / U{})> 
    {
        LAB_CHECK(d != 0 && !math::is_nan(d));
        return {x / d, y / d};
    }
    template <typename U>
    LAB_CPU_GPU Child<T> &operator/=(U d) 
    {
        LAB_CHECK_NE(d, 0);
        LAB_CHECK(!math::is_nan(d));
        x /= d;
        y /= d;
        return static_cast<Child<T> &>(*this);
    }

    LAB_CPU_GPU
    Child<T> operator-() const { return {-x, -y}; }

    LAB_CPU_GPU
    T operator[](int i) const 
    {
        LAB_CHECK(i >= 0 && i <= 1);
        return (i == 0) ? x : y;
    }

    LAB_CPU_GPU
    T &operator[](int i) 
    {
        LAB_CHECK(i >= 0 && i <= 1);
        return (i == 0) ? x : y;
    }

    std::string to_string() const { return internal::to_string2(x, y); }

    // Tuple2 Public Members
    T x{}, y{};
};

// Tuple2 Inline Functions
template <template <class> class C, typename T, typename U>
LAB_CPU_GPU inline auto operator*(U s, Tuple2<C, T> t) -> C<decltype(T{} * U{})> 
{
    LAB_CHECK(!t.has_nan());
    return t * s;
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline C<T> abs(Tuple2<C, T> t) 
{
    // "argument-dependent lookup..." (here and elsewhere)
    using std::abs;
    return {abs(t.x), abs(t.y)};
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline C<T> ceil(Tuple2<C, T> t) 
{
    using std::ceil;
    return {ceil(t.x), ceil(t.y)};
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline C<T> floor(Tuple2<C, T> t) 
{
    using std::floor;
    return {floor(t.x), floor(t.y)};
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline auto lerp(float t, Tuple2<C, T> t0, Tuple2<C, T> t1) 
{
    return (1 - t) * t0 + t * t1;
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline C<T> fma(float a, Tuple2<C, T> b, Tuple2<C, T> c) 
{
    return {math::fma(a, b.x, c.x), math::fma(a, b.y, c.y)};
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline C<T> fma(Tuple2<C, T> a, float b, Tuple2<C, T> c) 
{
    return fma(b, a, c);
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline C<T> min(Tuple2<C, T> t0, Tuple2<C, T> t1) 
{
    using std::min;
    return {min(t0.x, t1.x), min(t0.y, t1.y)};
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline T min_component_value(Tuple2<C, T> t) 
{
    using std::min;
    return min({t.x, t.y});
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline int min_component_index(Tuple2<C, T> t) 
{
    return (t.x < t.y) ? 0 : 1;
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline C<T> max(Tuple2<C, T> t0, Tuple2<C, T> t1) 
{
    using std::max;
    return {max(t0.x, t1.x), max(t0.y, t1.y)};
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline T max_component_value(Tuple2<C, T> t) 
{
    using std::max;
    return max({t.x, t.y});
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline int max_component_index(Tuple2<C, T> t) 
{
    return (t.x > t.y) ? 0 : 1;
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline C<T> permute(Tuple2<C, T> t, std::array<int, 2> p) 
{
    return {t[p[0]], t[p[1]]};
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline T hprod(Tuple2<C, T> t) 
{
    return t.x * t.y;
}

// Tuple3 Definition
template <template <typename> class Child, typename T>
class Tuple3 
{
public:
    // Tuple3 Public Methods
    Tuple3() = default;
    LAB_CPU_GPU
    Tuple3(T x, T y, T z) : x(x), y(y), z(z) { LAB_CHECK(!has_nan()); }

    LAB_CPU_GPU
    bool has_nan() const { return math::is_nan(x) || math::is_nan(y) || math::is_nan(z); }

    LAB_CPU_GPU
    T operator[](int i) const 
    {
        LAB_CHECK(i >= 0 && i <= 2);
        if (i == 0)
            return x;
        if (i == 1)
            return y;
        return z;
    }

    LAB_CPU_GPU
    T &operator[](int i) 
    {
        LAB_CHECK(i >= 0 && i <= 2);
        if (i == 0)
            return x;
        if (i == 1)
            return y;
        return z;
    }

    template <typename U>
    LAB_CPU_GPU auto operator+(Child<U> c) const -> Child<decltype(T{} + U{})> 
    {
        LAB_CHECK(!c.has_nan());
        return {x + c.x, y + c.y, z + c.z};
    }

    static const int n_dimensions = 3;

#ifdef LAB_DEBUG
    LAB_CPU_GPU
    Tuple3(Child<T> c) 
    {
        LAB_CHECK(!c.has_nan());
        x = c.x;
        y = c.y;
        z = c.z;
    }

    LAB_CPU_GPU
    Child<T> &operator=(Child<T> c) 
    {
        LAB_CHECK(!c.has_nan());
        x = c.x;
        y = c.y;
        z = c.z;
        return static_cast<Child<T> &>(*this);
    }
#endif

    template <typename U>
    LAB_CPU_GPU Child<T> &operator+=(Child<U> c) 
    {
        LAB_CHECK(!c.has_nan());
        x += c.x;
        y += c.y;
        z += c.z;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    LAB_CPU_GPU auto operator-(Child<U> c) const -> Child<decltype(T{} - U{})> 
    {
        LAB_CHECK(!c.has_nan());
        return {x - c.x, y - c.y, z - c.z};
    }
    template <typename U>
    LAB_CPU_GPU Child<T> &operator-=(Child<U> c) 
    {
        LAB_CHECK(!c.has_nan());
        x -= c.x;
        y -= c.y;
        z -= c.z;
        return static_cast<Child<T> &>(*this);
    }

    LAB_CPU_GPU
    bool operator==(Child<T> c) const { return x == c.x && y == c.y && z == c.z; }
    LAB_CPU_GPU
    bool operator!=(Child<T> c) const { return x != c.x || y != c.y || z != c.z; }

    template <typename U>
    LAB_CPU_GPU auto operator*(U s) const -> Child<decltype(T{} * U{})> 
    {
        return {s * x, s * y, s * z};
    }
    template <typename U>
    LAB_CPU_GPU Child<T> &operator*=(U s) 
    {
        LAB_CHECK(!math::is_nan(s));
        x *= s;
        y *= s;
        z *= s;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    LAB_CPU_GPU auto operator/(U d) const -> Child<decltype(T{} / U{})> 
    {
        LAB_CHECK_NE(d, 0);
        return {x / d, y / d, z / d};
    }
    template <typename U>
    LAB_CPU_GPU Child<T> &operator/=(U d) 
    {
        LAB_CHECK_NE(d, 0);
        x /= d;
        y /= d;
        z /= d;
        return static_cast<Child<T> &>(*this);
    }
    LAB_CPU_GPU
    Child<T> operator-() const { return {-x, -y, -z}; }

    std::string to_string() const { return internal::to_string3(x, y, z); }

    // Tuple3 Public Members
    T x{}, y{}, z{};
};

// Tuple3 Inline Functions
template <template <class> class C, typename T, typename U>
LAB_CPU_GPU inline auto operator*(U s, Tuple3<C, T> t) -> C<decltype(T{} * U{})> {
    return t * s;
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline C<T> abs(Tuple3<C, T> t) {
    using std::abs;
    return {abs(t.x), abs(t.y), abs(t.z)};
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline C<T> ceil(Tuple3<C, T> t) {
    using std::ceil;
    return {ceil(t.x), ceil(t.y), ceil(t.z)};
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline C<T> floor(Tuple3<C, T> t) {
    using std::floor;
    return {floor(t.x), floor(t.y), floor(t.z)};
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline auto lerp(float t, Tuple3<C, T> t0, Tuple3<C, T> t1) {
    return (1 - t) * t0 + t * t1;
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline C<T> fma(float a, Tuple3<C, T> b, Tuple3<C, T> c) {
    return {fma(a, b.x, c.x), fma(a, b.y, c.y), fma(a, b.z, c.z)};
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline C<T> fma(Tuple3<C, T> a, float b, Tuple3<C, T> c) {
    return fma(b, a, c);
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline C<T> min(Tuple3<C, T> t1, Tuple3<C, T> t2) {
    using std::min;
    return {min(t1.x, t2.x), min(t1.y, t2.y), min(t1.z, t2.z)};
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline T min_component_value(Tuple3<C, T> t) {
    using std::min;
    return min({t.x, t.y, t.z});
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline int min_component_index(Tuple3<C, T> t) {
    return (t.x < t.y) ? ((t.x < t.z) ? 0 : 2) : ((t.y < t.z) ? 1 : 2);
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline C<T> max(Tuple3<C, T> t1, Tuple3<C, T> t2) {
    using std::max;
    return {max(t1.x, t2.x), max(t1.y, t2.y), max(t1.z, t2.z)};
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline T max_component_value(Tuple3<C, T> t) {
    using std::max;
    return max({t.x, t.y, t.z});
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline int max_component_index(Tuple3<C, T> t) {
    return (t.x > t.y) ? ((t.x > t.z) ? 0 : 2) : ((t.y > t.z) ? 1 : 2);
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline C<T> permute(Tuple3<C, T> t, std::array<int, 3> p) {
    return {t[p[0]], t[p[1]], t[p[2]]};
}

template <template <class> class C, typename T>
LAB_CPU_GPU inline T hprod(Tuple3<C, T> t) {
    return t.x * t.y * t.z;
}

// Vector2 Definition
template <typename T>
class Vector2 : public Tuple2<Vector2, T> 
{
public:
    // Vector2 Public Methods
    using Tuple2<Vector2, T>::x;
    using Tuple2<Vector2, T>::y;

    Vector2() = default;
    LAB_CPU_GPU
    Vector2(T x, T y) : Tuple2<Vector2, T>(x, y) {}
    template <typename U>
    LAB_CPU_GPU explicit Vector2(math::Point2<U> p);
    template <typename U>
    LAB_CPU_GPU explicit Vector2(Vector2<U> v)
        : Tuple2<Vector2, T>(T(v.x), T(v.y)) {}
};

// Vector3 Definition
template <typename T>
class Vector3 : public Tuple3<Vector3, T> 
{
public:
    // Vector3 Public Methods
    using Tuple3<Vector3, T>::x;
    using Tuple3<Vector3, T>::y;
    using Tuple3<Vector3, T>::z;

    Vector3() = default;
    LAB_CPU_GPU
    Vector3(T x, T y, T z) : Tuple3<Vector3, T>(x, y, z) {}

    template <typename U>
    LAB_CPU_GPU explicit Vector3(Vector3<U> v)
        : Tuple3<Vector3, T>(T(v.x), T(v.y), T(v.z)) {}

    template <typename U>
    LAB_CPU_GPU explicit Vector3(math::Point3<U> p);
    template <typename U>
    LAB_CPU_GPU explicit Vector3(math::Normal3<U> n);
};

// Vector2* Definitions
using Vector2d = Vector2<double>;
using Vector2f = Vector2<float>;
using Vector2i = Vector2<int>;

// Vector3* Definitions
using Vector3d = Vector3<double>;
using Vector3f = Vector3<float>;
using Vector3i = Vector3<int>;

// Point2 Definition
template <typename T>
class Point2 : public Tuple2<Point2, T> 
{
public:
    // Point2 Public Methods
    using Tuple2<Point2, T>::x;
    using Tuple2<Point2, T>::y;
    using Tuple2<Point2, T>::has_nan;
    using Tuple2<Point2, T>::operator+;
    using Tuple2<Point2, T>::operator+=;
    using Tuple2<Point2, T>::operator*;
    using Tuple2<Point2, T>::operator*=;

    LAB_CPU_GPU
    Point2() { x = y = 0; }
    LAB_CPU_GPU
    Point2(T x, T y) : Tuple2<Point2, T>(x, y) {}
    template <typename U>
    LAB_CPU_GPU explicit Point2(Point2<U> v) : Tuple2<Point2, T>(T(v.x), T(v.y)) {}
    template <typename U>
    LAB_CPU_GPU explicit Point2(Vector2<U> v)
        : Tuple2<Point2, T>(T(v.x), T(v.y)) {}

    template <typename U>
    LAB_CPU_GPU auto operator+(Vector2<U> v) const -> Point2<decltype(T{} + U{})> 
    {
        LAB_CHECK(!v.has_nan());
        return {x + v.x, y + v.y};
    }
    template <typename U>
    LAB_CPU_GPU Point2<T> &operator+=(Vector2<U> v) 
    {
        LAB_CHECK(!v.has_nan());
        x += v.x;
        y += v.y;
        return *this;
    }

    LAB_CPU_GPU
    Point2<T> operator-() const { return {-x, -y}; }

    template <typename U>
    LAB_CPU_GPU auto operator-(Point2<U> p) const -> Vector2<decltype(T{} - U{})> 
    {
        LAB_CHECK(!p.has_nan());
        return {x - p.x, y - p.y};
    }
    template <typename U>
    LAB_CPU_GPU auto operator-(Vector2<U> v) const -> Point2<decltype(T{} - U{})> 
    {
        LAB_CHECK(!v.has_nan());
        return {x - v.x, y - v.y};
    }
    template <typename U>
    LAB_CPU_GPU Point2<T> &operator-=(Vector2<U> v) 
    {
        LAB_CHECK(!v.has_nan());
        x -= v.x;
        y -= v.y;
        return *this;
    }
};

// Point3 Definition
template <typename T>
class Point3 : public Tuple3<Point3, T> 
{
public:
    // Point3 Public Methods
    using Tuple3<Point3, T>::x;
    using Tuple3<Point3, T>::y;
    using Tuple3<Point3, T>::z;
    using Tuple3<Point3, T>::has_nan;
    using Tuple3<Point3, T>::operator+;
    using Tuple3<Point3, T>::operator+=;
    using Tuple3<Point3, T>::operator*;
    using Tuple3<Point3, T>::operator*=;

    Point3() = default;
    LAB_CPU_GPU
    Point3(T x, T y, T z) : Tuple3<Point3, T>(x, y, z) {}

    LAB_CPU_GPU
    Point3<T> operator-() const { return {-x, -y, -z}; }

    template <typename U>
    LAB_CPU_GPU explicit Point3(Point3<U> p)
        : Tuple3<Point3, T>(T(p.x), T(p.y), T(p.z)) {}
    template <typename U>
    LAB_CPU_GPU explicit Point3(Vector3<U> v)
        : Tuple3<Point3, T>(T(v.x), T(v.y), T(v.z)) {}

    template <typename U>
    LAB_CPU_GPU auto operator+(Vector3<U> v) const -> Point3<decltype(T{} + U{})> 
    {
        LAB_CHECK(!v.has_nan());
        return {x + v.x, y + v.y, z + v.z};
    }
    template <typename U>
    LAB_CPU_GPU Point3<T> &operator+=(Vector3<U> v) 
    {
        LAB_CHECK(!v.has_nan());
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    template <typename U>
    LAB_CPU_GPU auto operator-(Vector3<U> v) const -> Point3<decltype(T{} - U{})> 
    {
        LAB_CHECK(!v.has_nan());
        return {x - v.x, y - v.y, z - v.z};
    }
    template <typename U>
    LAB_CPU_GPU Point3<T> &operator-=(Vector3<U> v) 
    {
        LAB_CHECK(!v.has_nan());
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    template <typename U>
    LAB_CPU_GPU auto operator-(Point3<U> p) const -> Vector3<decltype(T{} - U{})> 
    {
        LAB_CHECK(!p.has_nan());
        return {x - p.x, y - p.y, z - p.z};
    }
};

// Point2* Definitions
using Point2d = Point2<double>;
using Point2f = Point2<float>;
using Point2i = Point2<int>;

// Point3* Definitions
using Point3d = Point3<double>;
using Point3f = Point3<float>;
using Point3i = Point3<int>;

// Normal3 Definition
template <typename T>
class Normal3 : public Tuple3<Normal3, T> 
{
public:
    // Normal3 Public Methods
    using Tuple3<Normal3, T>::x;
    using Tuple3<Normal3, T>::y;
    using Tuple3<Normal3, T>::z;
    using Tuple3<Normal3, T>::has_nan;
    using Tuple3<Normal3, T>::operator+;
    using Tuple3<Normal3, T>::operator*;
    using Tuple3<Normal3, T>::operator*=;

    Normal3() = default;
    LAB_CPU_GPU
    Normal3(T x, T y, T z) : Tuple3<Normal3, T>(x, y, z) {}
    template <typename U>
    LAB_CPU_GPU explicit Normal3<T>(Normal3<U> v)
        : Tuple3<Normal3, T>(T(v.x), T(v.y), T(v.z)) {}

    template <typename U>
    LAB_CPU_GPU explicit Normal3<T>(Vector3<U> v)
        : Tuple3<Normal3, T>(T(v.x), T(v.y), T(v.z)) {}
};

using Normal3d = Normal3<double>;
using Normal3f = Normal3<float>;

// Quaternion Definition
template<typename T>
class Quaternion 
{
public:
    // Quaternion Public Methods
    Quaternion() = default;

    LAB_CPU_GPU Quaternion(T a, T b, T c, T w)
        : v(a, b, c), w(w) {}

    LAB_CPU_GPU Quaternion(const Vector3<T>& v)
        : v(v) {}

    LAB_CPU_GPU
    Quaternion &operator+=(Quaternion q) 
    {
        v += q.v;
        w += q.w;
        return *this;
    }

    LAB_CPU_GPU
    Quaternion operator+(Quaternion q) const { return {v + q.v, w + q.w}; }
    LAB_CPU_GPU
    Quaternion &operator-=(Quaternion q) 
    {
        v -= q.v;
        w -= q.w;
        return *this;
    }
    LAB_CPU_GPU
    Quaternion operator-() const { return {-v, -w}; }
    LAB_CPU_GPU
    Quaternion operator-(Quaternion q) const { return {v - q.v, w - q.w}; }
    LAB_CPU_GPU
    Quaternion &operator*=(T f) 
    {
        v *= f;
        w *= f;
        return *this;
    }
    LAB_CPU_GPU
    Quaternion operator*(T f) const { return {v * f, w * f}; }
    LAB_CPU_GPU
    Quaternion &operator/=(T f) 
    {
        LAB_CHECK_NE(0, f);
        v /= f;
        w /= f;
        return *this;
    }
    LAB_CPU_GPU
    Quaternion operator/(T f) const 
    {
        LAB_CHECK_NE(0, f);
        return {v / f, w / f};
    }

    std::string to_string() const;

    Vector3<T> v;
    T w = 1;
};

using Quaterniond = Quaternion<double>;
using Quaternionf = Quaternion<float>;

template <typename T>
template <typename U>
Vector2<T>::Vector2(Point2<U> p) : Tuple2<Vector2, T>(T(p.x), T(p.y)) {}

template <typename T>
LAB_CPU_GPU inline auto dot(Vector2<T> v1, Vector2<T> v2) -> typename TupleLength<T>::type 
{
    LAB_CHECK(!v1.has_nan() && !v2.has_nan());
    return math::sum_of_products(v1.x, v2.x, v1.y, v2.y);
}

template <typename T>
LAB_CPU_GPU inline auto abs_dot(Vector2<T> v1, Vector2<T> v2) -> typename TupleLength<T>::type 
{
    LAB_CHECK(!v1.has_nan() && !v2.has_nan());
    return std::abs(dot(v1, v2));
}

template <typename T>
LAB_CPU_GPU inline auto length_squared(Vector2<T> v) -> typename TupleLength<T>::type 
{
    return Sqr(v.x) + Sqr(v.y);
}

template <typename T>
LAB_CPU_GPU inline auto length(Vector2<T> v) -> typename TupleLength<T>::type 
{
    using std::sqrt;
    return sqrt(length_squared(v));
}

template <typename T>
LAB_CPU_GPU inline auto normalize(Vector2<T> v) 
{
    return v / Length(v);
}

template <typename T>
LAB_CPU_GPU inline auto distance(Point2<T> p1, Point2<T> p2) -> typename TupleLength<T>::type 
{
    return length(p1 - p2);
}

template <typename T>
LAB_CPU_GPU inline auto distance_squared(Point2<T> p1, Point2<T> p2) -> typename TupleLength<T>::type 
{
    return length_squared(p1 - p2);
}

// Vector3 Inline Functions
template <typename T>
template <typename U>
Vector3<T>::Vector3(Point3<U> p) : Tuple3<Vector3, T>(T(p.x), T(p.y), T(p.z)) {}

template <typename T>
LAB_CPU_GPU inline Vector3<T> cross(Vector3<T> v1, Normal3<T> v2) 
{
    LAB_CHECK(!v1.has_nan() && !v2.has_nan());
    return {math::difference_of_products(v1.y, v2.z, v1.z, v2.y),
            math::difference_of_products(v1.z, v2.x, v1.x, v2.z),
            math::difference_of_products(v1.x, v2.y, v1.y, v2.x)};
}

template <typename T>
LAB_CPU_GPU inline Vector3<T> cross(Normal3<T> v1, Vector3<T> v2) 
{
    LAB_CHECK(!v1.has_nan() && !v2.has_nan());
    return {math::difference_of_products(v1.y, v2.z, v1.z, v2.y),
            math::difference_of_products(v1.z, v2.x, v1.x, v2.z),
            math::difference_of_products(v1.x, v2.y, v1.y, v2.x)};
}

template <typename T>
LAB_CPU_GPU inline T length_squared(Vector3<T> v) 
{
    return math::sqr(v.x) + math::sqr(v.y) + math::sqr(v.z);
}

template <typename T>
LAB_CPU_GPU inline auto length(Vector3<T> v) -> typename TupleLength<T>::type 
{
    using std::sqrt;
    return sqrt(length_squared(v));
}

template <typename T>
LAB_CPU_GPU inline auto normalize(Vector3<T> v) 
{
    return v / length(v);
}

template <typename T>
LAB_CPU_GPU inline T dot(Vector3<T> v, Vector3<T> w) 
{
    LAB_CHECK(!v.has_nan() && !w.has_nan());
    return v.x * w.x + v.y * w.y + v.z * w.z;
}

// Equivalent to std::acos(dot(a, b)), but more numerically stable.
// via http://www.plunk.org/~hatch/rightway.html
template <typename T>
LAB_CPU_GPU inline float angle_between(Vector3<T> v1, Vector3<T> v2) 
{
    if (dot(v1, v2) < 0)
        return math::Pi - 2 * math::safe_asin(length(v1 + v2) / 2);
    else
        return 2 * math::safe_asin(length(v2 - v1) / 2);
}

template <typename T>
LAB_CPU_GPU inline T abs_dot(Vector3<T> v1, Vector3<T> v2) 
{
    LAB_CHECK(!v1.has_nan() && !v2.has_nan());
    return std::abs(dot(v1, v2));
}

template <typename T>
LAB_CPU_GPU inline float angle_between(Normal3<T> a, Normal3<T> b) 
{
    if (dot(a, b) < 0)
        return math::Pi - 2 * math::safe_asin(length(a + b) / 2);
    else
        return 2 * math::safe_asin(length(b - a) / 2);
}

template <typename T>
LAB_CPU_GPU inline Vector3<T> gram_schmidt(Vector3<T> v, Vector3<T> w) 
{
    return v - dot(v, w) * w;
}

template <typename T>
LAB_CPU_GPU inline Vector3<T> cross(Vector3<T> v, Vector3<T> w) 
{
    LAB_CHECK(!v.has_nan() && !w.has_nan());
    return { math::difference_of_products(v.y, w.z, v.z, w.y), math::difference_of_products(v.z, w.x, v.x, w.z), math::difference_of_products(v.x, w.y, v.y, w.x) };
}

template <typename T>
LAB_CPU_GPU inline void coordinate_system(Vector3<T> v1, Vector3<T> *v2, Vector3<T> *v3) 
{
    float sign = std::copysign(float(1), v1.z);
    float a = -1 / (sign + v1.z);
    float b = v1.x * v1.y * a;
    *v2 = Vector3<T>(1 + sign * math::sqr(v1.x) * a, sign * b, -sign * v1.x);
    *v3 = Vector3<T>(b, sign + math::sqr(v1.y) * a, -v1.y);
}

template <typename T>
LAB_CPU_GPU inline void coordinate_system(Normal3<T> v1, Vector3<T> *v2, Vector3<T> *v3) 
{
    float sign = std::copysign(float(1), v1.z);
    float a = -1 / (sign + v1.z);
    float b = v1.x * v1.y * a;
    *v2 = Vector3<T>(1 + sign * math::sqr(v1.x) * a, sign * b, -sign * v1.x);
    *v3 = Vector3<T>(b, sign + math::sqr(v1.y) * a, -v1.y);
}

template <typename T>
template <typename U>
Vector3<T>::Vector3(Normal3<U> n) : Tuple3<Vector3, T>(T(n.x), T(n.y), T(n.z)) {}

// Point3 Inline Functions
template <typename T>
LAB_CPU_GPU inline auto distance(Point3<T> p1, Point3<T> p2) 
{
    return length(p1 - p2);
}

template <typename T>
LAB_CPU_GPU inline auto distance_squared(Point3<T> p1, Point3<T> p2) 
{
    return length_squared(p1 - p2);
}

// Normal3 Inline Functions
template <typename T>
LAB_CPU_GPU inline auto length_squared(Normal3<T> n) -> typename TupleLength<T>::type 
{
    return math::sqr(n.x) + math::sqr(n.y) + math::sqr(n.z);
}

template <typename T>
LAB_CPU_GPU inline auto Length(Normal3<T> n) -> typename TupleLength<T>::type 
{
    using std::sqrt;
    return sqrt(length_squared(n));
}

template <typename T>
LAB_CPU_GPU inline auto normalize(Normal3<T> n) 
{
    return n / length(n);
}

template <typename T>
LAB_CPU_GPU inline auto dot(Normal3<T> n, Vector3<T> v) -> typename TupleLength<T>::type 
{
    LAB_CHECK(!n.has_nan() && !v.has_nan());
    return math::fma(n.x, v.x, math::sum_of_products(n.y, v.y, n.z, v.z));
}

template <typename T>
LAB_CPU_GPU inline auto dot(Vector3<T> v, Normal3<T> n) -> typename TupleLength<T>::type 
{
    LAB_CHECK(!v.has_nan() && !n.has_nan());
    return math::fma(n.x, v.x, math::sum_of_products(n.y, v.y, n.z, v.z));
}

template <typename T>
LAB_CPU_GPU inline auto dot(Normal3<T> n1, Normal3<T> n2) -> typename TupleLength<T>::type 
{
    LAB_CHECK(!n1.has_nan() && !n2.has_nan());
    return math::fma(n1.x, n2.x, math::sum_of_products(n1.y, n2.y, n1.z, n2.z));
}

template <typename T>
LAB_CPU_GPU inline auto abs_dot(Normal3<T> n, Vector3<T> v) -> typename TupleLength<T>::type 
{
    LAB_CHECK(!n.has_nan() && !v.has_nan());
    return std::abs(dot(n, v));
}

template <typename T>
LAB_CPU_GPU inline auto abs_dot(Vector3<T> v, Normal3<T> n) -> typename TupleLength<T>::type 
{
    using std::abs;
    LAB_CHECK(!v.has_nan() && !n.has_nan());
    return abs(dot(v, n));
}

template <typename T>
LAB_CPU_GPU inline auto abs_dot(Normal3<T> n1, Normal3<T> n2) -> typename TupleLength<T>::type 
{
    using std::abs;
    LAB_CHECK(!n1.has_nan() && !n2.has_nan());
    return abs(dot(n1, n2));
}

template <typename T>
LAB_CPU_GPU inline Normal3<T> face_forward(Normal3<T> n, Vector3<T> v) 
{
    return (dot(n, v) < 0.f) ? -n : n;
}

template <typename T>
LAB_CPU_GPU inline Normal3<T> face_forward(Normal3<T> n, Normal3<T> n2) 
{
    return (dot(n, n2) < 0.f) ? -n : n;
}

template <typename T>
LAB_CPU_GPU inline Vector3<T> face_forward(Vector3<T> v, Vector3<T> v2) 
{
    return (dot(v, v2) < 0.f) ? -v : v;
}

template <typename T>
LAB_CPU_GPU inline Vector3<T> face_forward(Vector3<T> v, Normal3<T> n2) 
{
    return (dot(v, n2) < 0.f) ? -v : v;
}

// Quaternion Inline Functions
template<typename T, std::enable_if_t<std::is_floating_point<T>::value ,int> = 0>
LAB_CPU_GPU inline Quaternion<T> operator*(T f, Quaternion<T> q) 
{
    return q * f;
}

template<typename T, std::enable_if_t<std::is_floating_point<T>::value ,int> = 0>
LAB_CPU_GPU inline float dot(Quaternion<T> q1, Quaternion<T> q2) 
{
    return dot(q1.v, q2.v) + q1.w * q2.w;
}

template<typename T, std::enable_if_t<std::is_floating_point<T>::value ,int> = 0>
LAB_CPU_GPU inline float length(Quaternion<T> q) 
{
    return std::sqrt(dot(q, q));
}

template<typename T, std::enable_if_t<std::is_floating_point<T>::value ,int> = 0>
LAB_CPU_GPU inline Quaternion<T> normalize(Quaternion<T> q) 
{
    LAB_CHECK_GT(length(q), 0);
    return q / length(q);
}

template<typename T, std::enable_if_t<std::is_floating_point<T>::value ,int> = 0>
LAB_CPU_GPU inline T angle_between(Quaternion<T> q1, Quaternion<T> q2) 
{
    if (dot(q1, q2) < 0)
        return math::Pi - 2 * math::safe_asin(length(q1 + q2) / 2);
    else
        return 2 * math::safe_asin(length(q2 - q1) / 2);
}

template<typename T, std::enable_if_t<std::is_floating_point<T>::value ,int> = 0>
LAB_CPU_GPU inline Quaternion<T> slerp(T t, Quaternion<T> q1, Quaternion<T> q2) 
{
    T theta = angle_between(q1, q2);
    T sin_theta_over_theta = math::sin_x_over_x(theta);
    return q1 * (1 - t) * math::sin_x_over_x((1 - t) * theta) / sin_theta_over_theta +
           q2 * t * math::sin_x_over_x(t * theta) / sin_theta_over_theta;
}

// Bounds2 Definition
template <typename T>
class Bounds2 
{
public:
    // Bounds2 Public Methods
    LAB_CPU_GPU
    Bounds2() 
    {
        T minNum = std::numeric_limits<T>::lowest();
        T maxNum = std::numeric_limits<T>::max();
        pmin = Point2<T>(maxNum, maxNum);
        pmax = Point2<T>(minNum, minNum);
    }
    LAB_CPU_GPU
    explicit Bounds2(Point2<T> p) : pmin(p), pmax(p) {}
    LAB_CPU_GPU
    Bounds2(Point2<T> p1, Point2<T> p2) : pmin(min(p1, p2)), pmax(max(p1, p2)) {}
    template <typename U>
    LAB_CPU_GPU explicit Bounds2(const Bounds2<U> &b) 
    {
        if (b.is_empty())
            // Be careful about overflowing float->int conversions and the
            // like.
            *this = Bounds2<T>();
        else 
        {
            pmin = Point2<T>(b.pmin);
            pmax = Point2<T>(b.pmax);
        }
    }

    LAB_CPU_GPU
    Vector2<T> diagonal() const { return pmax - pmin; }

    LAB_CPU_GPU
    T area() const 
    {
        Vector2<T> d = pmax - pmin;
        return d.x * d.y;
    }

    LAB_CPU_GPU
    bool is_empty() const { return pmin.x >= pmax.x || pmin.y >= pmax.y; }

    LAB_CPU_GPU
    bool is_degenerate() const { return pmin.x > pmax.x || pmin.y > pmax.y; }

    LAB_CPU_GPU
    int max_dimension() const 
    {
        Vector2<T> diag = diagonal();
        if (diag.x > diag.y)
            return 0;
        else
            return 1;
    }
    LAB_CPU_GPU
    Point2<T> operator[](int i) const 
    {
        LAB_CHECK(i == 0 || i == 1);
        return (i == 0) ? pmin : pmax;
    }
    LAB_CPU_GPU
    Point2<T> &operator[](int i) 
    {
        LAB_CHECK(i == 0 || i == 1);
        return (i == 0) ? pmin : pmax;
    }
    LAB_CPU_GPU
    bool operator==(const Bounds2<T> &b) const 
    {
        return b.pmin == pmin && b.pmax == pmax;
    }
    LAB_CPU_GPU
    bool operator!=(const Bounds2<T> &b) const 
    {
        return b.pmin != pmin || b.pmax != pmax;
    }
    LAB_CPU_GPU
    Point2<T> corner(int corner) const 
    {
        LAB_CHECK(corner >= 0 && corner < 4);
        return Point2<T>((*this)[(corner & 1)].x, (*this)[(corner & 2) ? 1 : 0].y);
    }
    LAB_CPU_GPU
    Point2<T> lerp(Point2f t) const 
    {
        return Point2<T>(math::lerp(t.x, pmin.x, pmax.x),
                         math::lerp(t.y, pmin.y, pmax.y));
    }
    LAB_CPU_GPU
    Vector2<T> offset(Point2<T> p) const 
    {
        Vector2<T> o = p - pmin;
        if (pmax.x > pmin.x)
            o.x /= pmax.x - pmin.x;
        if (pmax.y > pmin.y)
            o.y /= pmax.y - pmin.y;
        return o;
    }
    LAB_CPU_GPU
    void bounding_sphere(Point2<T> *c, float *rad) const {
        *c = (pmin + pmax) / 2;
        *rad = inside(*c, *this) ? distance(*c, pmax) : 0;
    }

    std::string to_string() const { return string_printf("[ %s - %s ]", pmin, pmax); }

    // Bounds2 Public Members
    Point2<T> pmin, pmax;
};

// Bounds3 Definition
template <typename T>
class Bounds3 
{
public:
    // Bounds3 Public Methods
    LAB_CPU_GPU
    Bounds3() 
    {
        T minNum = std::numeric_limits<T>::lowest();
        T maxNum = std::numeric_limits<T>::max();
        pmin = Point3<T>(maxNum, maxNum, maxNum);
        pmax = Point3<T>(minNum, minNum, minNum);
    }

    LAB_CPU_GPU
    explicit Bounds3(Point3<T> p) : pmin(p), pmax(p) {}

    LAB_CPU_GPU
    Bounds3(Point3<T> p1, Point3<T> p2) : pmin(min(p1, p2)), pmax(max(p1, p2)) {}

    LAB_CPU_GPU
    Point3<T> operator[](int i) const 
    {
        LAB_CHECK(i == 0 || i == 1);
        return (i == 0) ? pmin : pmax;
    }
    LAB_CPU_GPU
    Point3<T> &operator[](int i) 
    {
        LAB_CHECK(i == 0 || i == 1);
        return (i == 0) ? pmin : pmax;
    }

    LAB_CPU_GPU
    Point3<T> corner(int corner) const 
    {
        LAB_CHECK(corner >= 0 && corner < 8);
        return Point3<T>((*this)[(corner & 1)].x, (*this)[(corner & 2) ? 1 : 0].y,
                         (*this)[(corner & 4) ? 1 : 0].z);
    }

    LAB_CPU_GPU
    Vector3<T> diagonal() const { return pmax - pmin; }

    LAB_CPU_GPU
    T surface_area() const 
    {
        Vector3<T> d = diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    LAB_CPU_GPU
    T volume() const 
    {
        Vector3<T> d = diagonal();
        return d.x * d.y * d.z;
    }

    LAB_CPU_GPU
    int max_dimension() const 
    {
        Vector3<T> d = diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }

    LAB_CPU_GPU
    Point3f lerp(Point3f t) const 
    {
        return Point3f(math::lerp(t.x, pmin.x, pmax.x), math::lerp(t.y, pmin.y, pmax.y),
                       math::lerp(t.z, pmin.z, pmax.z));
    }

    LAB_CPU_GPU
    Vector3f offset(Point3f p) const 
    {
        Vector3f o = p - pmin;
        if (pmax.x > pmin.x)
            o.x /= pmax.x - pmin.x;
        if (pmax.y > pmin.y)
            o.y /= pmax.y - pmin.y;
        if (pmax.z > pmin.z)
            o.z /= pmax.z - pmin.z;
        return o;
    }

    LAB_CPU_GPU
    void bounding_sphere(Point3<T> *center, float *radius) const 
    {
        *center = (pmin + pmax) / 2;
        *radius = inside(*center, *this) ? distance(*center, pmax) : 0;
    }

    LAB_CPU_GPU
    bool is_empty() const 
    {
        return pmin.x >= pmax.x || pmin.y >= pmax.y || pmin.z >= pmax.z;
    }
    LAB_CPU_GPU
    bool is_degenerate() const 
    {
        return pmin.x > pmax.x || pmin.y > pmax.y || pmin.z > pmax.z;
    }

    template <typename U>
    LAB_CPU_GPU explicit Bounds3(const Bounds3<U> &b) 
    {
        if (b.is_empty())
            // Be careful about overflowing float->int conversions and the
            // like.
            *this = Bounds3<T>();
        else {
            pmin = Point3<T>(b.pmin);
            pmax = Point3<T>(b.pmax);
        }
    }
    LAB_CPU_GPU
    bool operator==(const Bounds3<T> &b) const 
    {
        return b.pmin == pmin && b.pmax == pmax;
    }
    LAB_CPU_GPU
    bool operator!=(const Bounds3<T> &b) const 
    {
        return b.pmin != pmin || b.pmax != pmax;
    }
    LAB_CPU_GPU
    bool intersectP(Point3f o, Vector3f d, float tmax = math::Infinity, float *hitt0 = nullptr,
                    float *hitt1 = nullptr) const;
    LAB_CPU_GPU
    bool intersectP(Point3f o, Vector3f d, float tmax, Vector3f invDir,
                    const int dirIsNeg[3]) const;

    std::string to_string() const { return string_printf("[ %s - %s ]", pmin, pmax); }

    // Bounds3 Public Members
    Point3<T> pmin, pmax;
};

// Bounds[23][fi] Definitions
using Bounds2f = Bounds2<float>;
using Bounds2i = Bounds2<int>;
using Bounds3f = Bounds3<float>;
using Bounds3i = Bounds3<int>;

// Bounds2 Inline Functions
template <typename T>
LAB_CPU_GPU inline Bounds2<T> Union(const Bounds2<T> &b1, const Bounds2<T> &b2) 
{
    // Be careful to not run the two-point Bounds constructor.
    Bounds2<T> ret;
    ret.pmin = min(b1.pmin, b2.pmin);
    ret.pmax = max(b1.pmax, b2.pmax);
    return ret;
}

template <typename T>
LAB_CPU_GPU inline Bounds2<T> intersect(const Bounds2<T> &b1, const Bounds2<T> &b2) 
{
    // Be careful to not run the two-point Bounds constructor.
    Bounds2<T> b;
    b.pmin = max(b1.pmin, b2.pmin);
    b.pmax = min(b1.pmax, b2.pmax);
    return b;
}

template <typename T>
LAB_CPU_GPU inline bool overlaps(const Bounds2<T> &ba, const Bounds2<T> &bb) 
{
    bool x = (ba.pmax.x >= bb.pmin.x) && (ba.pmin.x <= bb.pmax.x);
    bool y = (ba.pmax.y >= bb.pmin.y) && (ba.pmin.y <= bb.pmax.y);
    return (x && y);
}

template <typename T>
LAB_CPU_GPU inline bool inside(Point2<T> pt, const Bounds2<T> &b) 
{
    return (pt.x >= b.pmin.x && pt.x <= b.pmax.x && pt.y >= b.pmin.y && pt.y <= b.pmax.y);
}

template <typename T>
LAB_CPU_GPU inline bool inside(const Bounds2<T> &ba, const Bounds2<T> &bb) 
{
    return (ba.pmin.x >= bb.pmin.x && ba.pmax.x <= bb.pmax.x && ba.pmin.y >= bb.pmin.y &&
            ba.pmax.y <= bb.pmax.y);
}

template <typename T>
LAB_CPU_GPU inline bool inside_exclusive(Point2<T> pt, const Bounds2<T> &b) 
{
    return (pt.x >= b.pmin.x && pt.x < b.pmax.x && pt.y >= b.pmin.y && pt.y < b.pmax.y);
}

template <typename T, typename U>
LAB_CPU_GPU inline Bounds2<T> expand(const Bounds2<T> &b, U delta) 
{
    Bounds2<T> ret;
    ret.pmin = b.pmin - Vector2<T>(delta, delta);
    ret.pmax = b.pmax + Vector2<T>(delta, delta);
    return ret;
}

// Bounds3 Inline Functions
template <typename T>
LAB_CPU_GPU inline Bounds3<T> Union(const Bounds3<T> &b, Point3<T> p) 
{
    Bounds3<T> ret;
    ret.pmin = min(b.pmin, p);
    ret.pmax = max(b.pmax, p);
    return ret;
}

template <typename T>
LAB_CPU_GPU inline Bounds3<T> Union(const Bounds3<T> &b1, const Bounds3<T> &b2) 
{
    Bounds3<T> ret;
    ret.pmin = min(b1.pmin, b2.pmin);
    ret.pmax = max(b1.pmax, b2.pmax);
    return ret;
}

template <typename T>
LAB_CPU_GPU inline Bounds3<T> intersect(const Bounds3<T> &b1, const Bounds3<T> &b2) 
{
    Bounds3<T> b;
    b.pmin = max(b1.pmin, b2.pmin);
    b.pmax = min(b1.pmax, b2.pmax);
    return b;
}

template <typename T>
LAB_CPU_GPU inline bool overlaps(const Bounds3<T> &b1, const Bounds3<T> &b2) 
{
    bool x = (b1.pmax.x >= b2.pmin.x) && (b1.pmin.x <= b2.pmax.x);
    bool y = (b1.pmax.y >= b2.pmin.y) && (b1.pmin.y <= b2.pmax.y);
    bool z = (b1.pmax.z >= b2.pmin.z) && (b1.pmin.z <= b2.pmax.z);
    return (x && y && z);
}

template <typename T>
LAB_CPU_GPU inline bool inside(Point3<T> p, const Bounds3<T> &b) 
{
    return (p.x >= b.pmin.x && p.x <= b.pmax.x && p.y >= b.pmin.y && p.y <= b.pmax.y &&
            p.z >= b.pmin.z && p.z <= b.pmax.z);
}

template <typename T>
LAB_CPU_GPU inline bool inside_exclusive(Point3<T> p, const Bounds3<T> &b) 
{
    return (p.x >= b.pmin.x && p.x < b.pmax.x && p.y >= b.pmin.y && p.y < b.pmax.y &&
            p.z >= b.pmin.z && p.z < b.pmax.z);
}

template <typename T, typename U>
LAB_CPU_GPU inline auto distance_squared(Point3<T> p, const Bounds3<U> &b) 
{
    using TDist = decltype(T{} - U{});
    TDist dx = std::max<TDist>({0, b.pmin.x - p.x, p.x - b.pmax.x});
    TDist dy = std::max<TDist>({0, b.pmin.y - p.y, p.y - b.pmax.y});
    TDist dz = std::max<TDist>({0, b.pmin.z - p.z, p.z - b.pmax.z});
    return math::sqr(dx) + math::sqr(dy) + math::sqr(dz);
}

template <typename T, typename U>
LAB_CPU_GPU inline auto distance(Point3<T> p, const Bounds3<U> &b) 
{
    auto dist2 = distance_squared(p, b);
    using TDist = typename TupleLength<decltype(dist2)>::type;
    return std::sqrt(TDist(dist2));
}

template <typename T, typename U>
LAB_CPU_GPU inline Bounds3<T> expand(const Bounds3<T> &b, U delta) 
{
    Bounds3<T> ret;
    ret.pmin = b.pmin - Vector3<T>(delta, delta, delta);
    ret.pmax = b.pmax + Vector3<T>(delta, delta, delta);
    return ret;
}

template <typename T>
LAB_CPU_GPU inline bool Bounds3<T>::intersectP(Point3f o, Vector3f d, float tmax, float *hitt0, float *hitt1) const 
{
    float t0 = 0, t1 = tmax;
    for (int i = 0; i < 3; ++i) 
    {
        // Update interval for _i_th bounding box slab
        float invRayDir = 1 / d[i];
        float tNear = (pmin[i] - o[i]) * invRayDir;
        float tFar = (pmax[i] - o[i]) * invRayDir;
        // Update parametric interval from slab intersection $t$ values
        if (tNear > tFar)
            std::swap(tNear, tFar);
        // Update _tFar_ to ensure robust ray--bounds intersection
        tFar *= 1 + 2 * gamma(3);

        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;
        if (t0 > t1)
            return false;
    }
    if (hitt0)
        *hitt0 = t0;
    if (hitt1)
        *hitt1 = t1;
    return true;
}

template <typename T>
LAB_CPU_GPU inline bool Bounds3<T>::intersectP(Point3f o, Vector3f d, float raytmax, Vector3f invDir, const int dirIsNeg[3]) const 
{
    const Bounds3f &bounds = *this;
    // Check for ray intersection against $x$ and $y$ slabs
    float tmin = (bounds[dirIsNeg[0]].x - o.x) * invDir.x;
    float tmax = (bounds[1 - dirIsNeg[0]].x - o.x) * invDir.x;
    float tymin = (bounds[dirIsNeg[1]].y - o.y) * invDir.y;
    float tymax = (bounds[1 - dirIsNeg[1]].y - o.y) * invDir.y;
    // Update _tmax_ and _tymax_ to ensure robust bounds intersection
    tmax *= 1 + 2 * gamma(3);
    tymax *= 1 + 2 * gamma(3);

    if (tmin > tymax || tymin > tmax)
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    // Check for ray intersection against $z$ slab
    float tzmin = (bounds[dirIsNeg[2]].z - o.z) * invDir.z;
    float tzmax = (bounds[1 - dirIsNeg[2]].z - o.z) * invDir.z;
    // Update _tzmax_ to ensure robust bounds intersection
    tzmax *= 1 + 2 * gamma(3);

    if (tmin > tzmax || tzmin > tmax)
        return false;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

    return (tmin < raytmax) && (tmax > 0);
}

template <typename T>
LAB_CPU_GPU inline Bounds2<T> Union(const Bounds2<T> &b, Point2<T> p) 
{
    // Be careful to not run the two-point Bounds constructor.
    Bounds2<T> ret;
    ret.pmin = min(b.pmin, p);
    ret.pmax = max(b.pmax, p);
    return ret;
}

// Spherical Geometry Inline Functions
LAB_CPU_GPU inline float spherical_triangle_area(Vector3f a, Vector3f b, Vector3f c) 
{
    return std::abs(
        2 * std::atan2(dot(a, cross(b, c)), 1 + dot(a, b) + dot(a, c) + dot(b, c)));
}

LAB_CPU_GPU inline float spherical_quad_area(Vector3f a, Vector3f b, Vector3f c, Vector3f d);

LAB_CPU_GPU inline float spherical_quad_area(Vector3f a, Vector3f b, Vector3f c, Vector3f d) 
{
    Vector3f axb = cross(a, b), bxc = cross(b, c);
    Vector3f cxd = cross(c, d), dxa = cross(d, a);
    if (length_squared(axb) == 0 || length_squared(bxc) == 0 || length_squared(cxd) == 0 ||
        length_squared(dxa) == 0)
        return 0;
    axb = normalize(axb);
    bxc = normalize(bxc);
    cxd = normalize(cxd);
    dxa = normalize(dxa);

    float alpha = angle_between(dxa, -axb);
    float beta = angle_between(axb, -bxc);
    float gamma = angle_between(bxc, -cxd);
    float delta = angle_between(cxd, -dxa);

    return std::abs(alpha + beta + gamma + delta - 2 * math::Pi);
}

LAB_CPU_GPU inline Vector3f spherical_direction(float sin_theta, float cos_theta, float phi) 
{
    LAB_CHECK(sin_theta >= -1.0001 && sin_theta <= 1.0001);
    LAB_CHECK(cos_theta >= -1.0001 && cos_theta <= 1.0001);
    return Vector3f(math::clamp(sin_theta, -1, 1) * std::cos(phi),
                    math::clamp(sin_theta, -1, 1) * std::sin(phi), math::clamp(cos_theta, -1, 1));
}

LAB_CPU_GPU inline float spherical_theta(Vector3f v) 
{
    return math::safe_acos(v.z);
}

LAB_CPU_GPU inline float SphericalPhi(Vector3f v) 
{
    float p = std::atan2(v.y, v.x);
    return (p < 0) ? (p + 2 * math::Pi) : p;
}

LAB_CPU_GPU inline float cos_theta(Vector3f w) 
{
    return w.z;
}
LAB_CPU_GPU inline float cos2_theta(Vector3f w) 
{
    return math::sqr(w.z);
}
LAB_CPU_GPU inline float abs_cos_theta(Vector3f w) 
{
    return std::abs(w.z);
}

LAB_CPU_GPU inline float sin2_theta(Vector3f w) 
{
    return std::max<float>(0, 1 - cos2_theta(w));
}
LAB_CPU_GPU inline float sin_theta(Vector3f w) 
{
    return std::sqrt(sin2_theta(w));
}

LAB_CPU_GPU inline float tan_theta(Vector3f w) 
{
    return sin_theta(w) / cos_theta(w);
}
LAB_CPU_GPU inline float tan2_theta(Vector3f w) 
{
    return sin2_theta(w) / cos2_theta(w);
}

LAB_CPU_GPU inline float cos_phi(Vector3f w) 
{
    float sintheta = sin_theta(w);
    return (sintheta == 0) ? 1 : math::clamp(w.x / sintheta, -1, 1);
}
LAB_CPU_GPU inline float sin_phi(Vector3f w) 
{
    float sintheta = sin_theta(w);
    return (sintheta == 0) ? 0 : math::clamp(w.y / sintheta, -1, 1);
}

LAB_CPU_GPU inline float cos_dphi(Vector3f wa, Vector3f wb) 
{
    float waxy = math::sqr(wa.x) + math::sqr(wa.y), wbxy = math::sqr(wb.x) + math::sqr(wb.y);
    if (waxy == 0 || wbxy == 0)
        return 1;
    return math::clamp((wa.x * wb.x + wa.y * wb.y) / std::sqrt(waxy * wbxy), -1, 1);
}

LAB_CPU_GPU inline bool same_hemisphere(Vector3f w, Vector3f wp) 
{
    return w.z * wp.z > 0;
}

LAB_CPU_GPU inline bool same_hemisphere(Vector3f w, Normal3f wp) 
{
    return w.z * wp.z > 0;
}


}

}