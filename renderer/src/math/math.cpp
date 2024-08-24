#include "renderer/math/math.h"
#include "renderer/utils/stringprint.h"


namespace lab
{

namespace math
{

std::string CompensatedDouble::to_string() const 
{
    return renderer::string_printf("[ CompensatedDouble v: %f err: %f ]", v, err);
}

template <typename Float>
std::string CompensatedSum<Float>::to_string() const 
{
    return renderer::string_printf("[ CompensatedSum sum: %s c: %s ]", sum, c);
}

}

}