#pragma once

#include "renderer/core.h"

namespace lab
{

namespace renderer
{

enum class EBodyType 
{ 
    Static, 
    Dynamic, 
    Kinematic 
};

enum class ECollisionDetectionType : uint32_t
{
    Discrete,
    Continuous
};

}

}