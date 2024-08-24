#include "renderer/scene/scene.h"


namespace lab
{

namespace renderer
{

Registry Scene::registry_;

Scene::Scene(const std::string& name)
    : name_(name) {}

Registry& Scene::get_registry() 
{
    return registry_;
}

std::string Scene::get_name() 
{
    return name_;
}


}

}