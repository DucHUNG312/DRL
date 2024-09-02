#pragma once

#include "renderer/core.h"

#include "renderer/utils/stringprint.h"
#include "renderer/utils/uuid.h"

#include "renderer/math/math.h"
#include "renderer/math/vec.h"
#include "renderer/math/mat.h"

#include "renderer/scene/register.h"
#include "renderer/scene/scene.h"

#ifdef LAB_GPU_BUILD
#include "renderer/cuda/core.h"
#endif

namespace lab
{

namespace render
{

class Renderer
{
public:
    static void init()
    {

    }

    static void render()
    {

    }

    static void shutdown()
    {

    }
};

}

}