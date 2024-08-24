#include "renderer/math/vec.h"
#include "renderer/utils/stringprint.h"

namespace lab
{

namespace math
{

template <typename T>
std::string internal::to_string2(T x, T y) 
{
    if (std::is_floating_point_v<T>)
        return renderer::string_printf("[ %f, %f ]", x, y);
    else
        return renderer::string_printf("[ %d, %d ]", x, y);
}

template <typename T>
std::string internal::to_string3(T x, T y, T z) 
{
    if (std::is_floating_point_v<T>)
        return renderer::string_printf("[ %f, %f, %f ]", x, y, z);
    else
        return renderer::string_printf("[ %d, %d, %d ]", x, y, z);
}

template std::string internal::to_string2(float, float);
template std::string internal::to_string2(double, double);
template std::string internal::to_string2(int, int);
template std::string internal::to_string3(float, float, float);
template std::string internal::to_string3(double, double, double);
template std::string internal::to_string3(int, int, int);

// Quaternion Method Definitions
template<typename T>
std::string Quaternion<T>::to_string() const 
{
    return renderer::string_printf("[ %f, %f, %f, %f ]", v.x, v.y, v.z, w);
}

}

}