#include "renderer/scene/components.h"

namespace lab
{

namespace renderer
{

IDComponent::IDComponent(const std::string& id) 
    : id(id) {}

NameComponent::NameComponent(const std::string& name) 
    : name(name) {}

TransformComponent::TransformComponent(const math::Vector3d& translation)
    : translation(translation) {}

math::Vector3d TransformComponent::get_roration_euler() const
{
    return roration_euler;
}

void TransformComponent::set_roration_euler(const math::Vector3d& euler)
{
    roration_euler = euler;
    rotation = math::Quaterniond(roration_euler);
}

math::Quaterniond TransformComponent::get_rotation() const
{
    return rotation;
}

ExternalForce::ExternalForce(double mag, bool active /*= true*/) 
    : magnitude(mag), active(active) {}

ExternalTorque::ExternalTorque(const math::Vector3d& axis, double magnitude, const math::Vector3d& application_point, bool active /*= true*/) 
    : axis(axis), magnitude(magnitude), application_point(application_point), active(active) {}

Velocity::Velocity(const math::Vector3d& linear, const math::Vector3d& angular, bool active /*= true*/) 
    : linear(linear), angular(angular), active(active) {}

RigidBody2DComponent::RigidBody2DComponent(double mass) 
    : mass(mass) {}

}

}