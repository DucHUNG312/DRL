#pragma once

#include "lab/core.h"

namespace lab
{
namespace utils
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

constexpr double Min = std::numeric_limits<double>::min();
constexpr double Max = std::numeric_limits<double>::max();

template <typename T>
class CompensatedSum 
{
public:
    CompensatedSum() = default;
    LAB_CPU_GPU explicit CompensatedSum(T v) : sum(v) {}

    LAB_CPU_GPU CompensatedSum &operator=(T v) 
    {
        sum = v;
        c = 0;
        return *this;
    }

    LAB_CPU_GPU CompensatedSum &operator+=(T v) 
    {
        T delta = v - c;
        T newSum = sum + delta;
        c = (newSum - sum) - delta;
        sum = newSum;
        return *this;
    }

    LAB_CPU_GPU explicit operator T() const { return sum; }
private:
    T sum = 0;
    T c = 0;
};

struct CompensatedDouble 
{
public:
    LAB_CPU_GPU CompensatedDouble(double v, double err = 0) : v(v), err(err) {}
    LAB_CPU_GPU explicit operator double() const { return v + err; }

    double v, err;
};

LAB_CPU_GPU inline double rad(double deg) 
{
    return (Pi / 180) * deg;
}
LAB_CPU_GPU inline double deg(double rad) 
{
    return (180 / Pi) * rad;
}

LAB_CPU_GPU inline double safe_sqrt(double x) 
{
    assert(x >= -1e-3);
    return std::sqrt(std::max(0., x));
}


}
}
}