#pragma once

#include "renderer/core.h"
#include "renderer/cuda/core.h"

namespace lab
{
namespace math
{

constexpr double Epsilon = 0.0001;
constexpr double Pi = 3.14159265358979323846;
constexpr double InvPi = 0.31830988618379067154;
constexpr double Inv2Pi = 0.15915494309189533577;
constexpr double Inv4Pi = 0.07957747154594766788;
constexpr double PiOver2 = 1.57079632679489661923;
constexpr double PiOver4 = 0.78539816339744830961;
constexpr double Sqrt2 = 1.41421356237309504880;
constexpr float Infinity = std::numeric_limits<float>::infinity();
constexpr float DInfinity = std::numeric_limits<double>::infinity();
const double Min = std::numeric_limits<double>::min();
const double Max = std::numeric_limits<double>::max();

template <typename T>
class CompensatedSum 
{
public:
    CompensatedSum() = default;
    explicit CompensatedSum(T v) : sum(v) {}

    CompensatedSum &operator=(T v) 
    {
        sum = v;
        c = 0;
        return *this;
    }

    CompensatedSum &operator+=(T v) 
    {
        T delta = v - c;
        T newSum = sum + delta;
        c = (newSum - sum) - delta;
        sum = newSum;
        return *this;
    }

    std::string to_string() const;

    explicit operator T() const { return sum; }
private:
    T sum = 0;
    T c = 0;
};

struct CompensatedDouble
{
public:
    CompensatedDouble(double v, double err = 0) : v(v), err(err) {}
    explicit operator double() const { return v + err; }

    std::string to_string() const;

    double v, err;
};

template <typename T>
LAB_CPU_GPU inline typename std::enable_if_t<std::is_floating_point<T>::value, bool> is_nan(T v) 
{
#ifdef LAB_IS_GPU_CODE
    return isnan(v);
#else
    return std::isnan(v);
#endif
}

template <typename T>
LAB_CPU_GPU inline typename std::enable_if_t<std::is_integral<T>::value, bool> is_nan(T v) 
{
    return false;
}

template <typename T>
LAB_CPU_GPU inline typename std::enable_if_t<std::is_floating_point<T>::value, bool> is_inf(T v) 
{
#ifdef LAB_IS_GPU_CODE
    return isinf(v);
#else
    return std::isinf(v);
#endif
}

template <typename T>
LAB_CPU_GPU inline typename std::enable_if_t<std::is_integral<T>::value, bool> is_inf(T v) 
{
    return false;
}

template <typename T>
LAB_CPU_GPU inline typename std::enable_if_t<std::is_floating_point<T>::value, bool> is_finite( T v) 
{
#ifdef LAB_IS_GPU_CODE
    return isfinite(v);
#else
    return std::isfinite(v);
#endif
}
template <typename T>
LAB_CPU_GPU inline typename std::enable_if_t<std::is_integral<T>::value, bool> is_finite(T v) 
{
    return true;
}

LAB_CPU_GPU inline double rad(double deg) 
{
    return (Pi / 180) * deg;
}
LAB_CPU_GPU inline double deg(double rad) 
{
    return (180 / Pi) * rad;
}

template <typename T>
LAB_CPU_GPU inline constexpr T sqr(T v) 
{
    return v * v;
}

template <typename T, typename U, typename V>
LAB_CPU_GPU inline constexpr T clamp(T val, U low, V high) 
{
    if (val < low)
        return T(low);
    else if (val > high)
        return T(high);
    else
        return val;
}

LAB_CPU_GPU inline double safe_sqrt(double x) 
{
    assert(x >= -1e-3);
    return std::sqrt(std::max(0., x));
}

LAB_CPU_GPU inline float lerp(float x, float a, float b) 
{
    return (1 - x) * a + x * b;
}

template <typename T>
LAB_CPU_GPU inline typename std::enable_if_t<std::is_integral<T>::value, T> fma(T a, T b, T c) 
{
    return a * b + c;
}

LAB_CPU_GPU inline float fma(float a, float b, float c) 
{
    return std::fma(a, b, c);
}

LAB_CPU_GPU inline double fma(double a, double b, double c) 
{
    return std::fma(a, b, c);
}

LAB_CPU_GPU inline long double fma(long double a, long double b, long double c) 
{
    return std::fma(a, b, c);
}

LAB_CPU_GPU inline float sin_x_over_x(float x) 
{
    if (1 - x * x == 1)
        return 1;
    return std::sin(x) / x;
}

LAB_CPU_GPU inline float safe_asin(float x) 
{
    LAB_CHECK(x >= -1.0001 && x <= 1.0001);
    return std::asin(clamp(x, -1, 1));
}
LAB_CPU_GPU inline float safe_acos(float x) 
{
    LAB_CHECK(x >= -1.0001 && x <= 1.0001);
    return std::acos(clamp(x, -1, 1));
}

LAB_CPU_GPU inline double safe_asin(double x) 
{
    LAB_CHECK(x >= -1.0001 && x <= 1.0001);
    return std::asin(clamp(x, -1, 1));
}

LAB_CPU_GPU inline double safe_acos(double x) 
{
    LAB_CHECK(x >= -1.0001 && x <= 1.0001);
    return std::acos(clamp(x, -1, 1));
}

LAB_CPU_GPU inline CompensatedDouble two_prod(double a, double b) 
{
    double ab = a * b;
    return {ab, fma(a, b, -ab)};
}

LAB_CPU_GPU inline CompensatedDouble two_sum(double a, double b) 
{
    double s = a + b, delta = s - a;
    return {s, (a - (s - delta)) + (b - delta)};
}

template <typename Ta, typename Tb, typename Tc, typename Td>
LAB_CPU_GPU inline auto difference_of_products(Ta a, Tb b, Tc c, Td d) 
{
    auto cd = c * d;
    auto dop = fma(a, b, -cd);
    auto error = fma(-c, d, cd);
    return dop + error;
}

template <typename Ta, typename Tb, typename Tc, typename Td>
LAB_CPU_GPU inline auto sum_of_products(Ta a, Tb b, Tc c, Td d) 
{
    auto cd = c * d;
    auto sop = fma(a, b, cd);
    auto error = fma(c, d, -cd);
    return sop + error;
}

namespace internal 
{
template <typename Float>
LAB_CPU_GPU inline CompensatedDouble inner_product(Float a, Float b) 
{
    return two_prod(a, b);
}

template <typename Float, typename... T>
LAB_CPU_GPU inline CompensatedDouble inner_product(Float a, Float b, T... terms) 
{
    CompensatedDouble ab = two_prod(a, b);
    CompensatedDouble tp = inner_product(terms...);
    CompensatedDouble sum = two_sum(ab.v, tp.v);
    return {sum.v, ab.err + (tp.err + sum.err)};
}

}  // namespace internal

template <typename... T>
LAB_CPU_GPU inline std::enable_if_t<std::conjunction_v<std::is_arithmetic<T>...>, double> inner_product(T... terms) 
{
    CompensatedDouble ip = internal::inner_product(terms...);
    return double(ip);
}

}
}