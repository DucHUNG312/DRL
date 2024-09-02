#pragma once

#include "renderer/core.h"
#include "renderer/scene/register.h"

namespace lab
{

namespace renderer
{

struct SceneSettings
{
    bool physics_enabled_2D = true;
    bool physics_enabled_3D = true;
};

class Scene
{
public:
    explicit Scene(const std::string& name);
    LAB_DEFAULT_CONSTRUCT(Scene);

    std::string get_name();

    static Registry& get_registry();
protected:
    std::string name_;
    static Registry registry_;
};

}

}