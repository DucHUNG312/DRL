#pragma once

#include "renderer/core.h"
#include "renderer/math/vec.h"
#include "renderer/physics/type.h"

#include <entt/entt.hpp>

namespace lab
{

namespace renderer
{

struct IDComponent
{
    std::string id;
public:
    explicit IDComponent(const std::string& id);
    LAB_DEFAULT_CONSTRUCT(IDComponent);
};

struct NameComponent
{
    std::string name;
public:
    explicit NameComponent(const std::string& name);
    LAB_DEFAULT_CONSTRUCT(NameComponent);
};

struct ParentComponent
{
    entt::entity parent = entt::null;
    LAB_DEFAULT_CONSTRUCT(ParentComponent);
};

struct ChildrenComponent
{
    std::list<entt::entity> children;
    LAB_DEFAULT_CONSTRUCT(ChildrenComponent);
};

struct RenderableComponent
{
    std::list<entt::entity> children;
    LAB_DEFAULT_CONSTRUCT(RenderableComponent);
};

struct TransformComponent
{
    math::Vector3d translation = { 0.0, 0.0, 0.0 };
    math::Vector3d scale = { 1.0, 1.0, 1.0 };
    math::Vector3d roration_euler = { 0.0, 0.0, 0.0 };
    math::Quaterniond rotation = {  1.0, 0.0, 0.0, 0.0 };

    TransformComponent(const math::Vector3d& translation);
    math::Vector3d get_roration_euler() const;
    math::Quaterniond get_rotation() const;
    void set_roration_euler(const math::Vector3d& euler);
    LAB_DEFAULT_CONSTRUCT(TransformComponent);
};

struct ExternalForce
{
    math::Vector3d direction;
    double magnitude = 0.01;
    bool active = false;

    LAB_DEFAULT_CONSTRUCT(ExternalForce);
    explicit ExternalForce(double mag, bool active = true);
};

struct ExternalTorque 
{
    math::Vector3d axis;
    double magnitude = 0.01;
    math::Vector3d application_point;
    bool active = false;

    LAB_DEFAULT_CONSTRUCT(ExternalTorque);
    explicit ExternalTorque(const math::Vector3d& axis, double magnitude, const math::Vector3d& application_point, bool active = true);
};

struct Velocity 
{
    math::Vector3d linear;
    math::Vector3d angular;
    bool active = false;

    LAB_DEFAULT_CONSTRUCT(Velocity);
    explicit Velocity(const math::Vector3d& linear, const math::Vector3d& angular, bool active = true);
};

struct RigidBody2DComponent
{
    enum class Type { None = -1, Static, Dynamic, Kinematic };
    Type body_type;
    bool fixed_rotation = false;
    double mass = 1.0;
    double linear_drag = 0.01;
    double angular_drag = 0.05;
    double gravity_scale = 1.0;
    bool is_bullet = false;
    bool active = false;

    LAB_DEFAULT_CONSTRUCT(RigidBody2DComponent);
    explicit RigidBody2DComponent(double mass);
};

struct BoxCollider2DComponent
{
    math::Vector2d offset = { 0.0f,0.0f };
    math::Vector2d size = { 0.5, 0.5 };
    double density = 1.0;
    double friction = 1.0;

    LAB_DEFAULT_CONSTRUCT(BoxCollider2DComponent);
};

struct CircleCollider2DComponent
{
    math::Vector2d offset = { 0.0f,0.0f };
    double radius = 1.0;
    double density = 1.0;
    double friction = 1.0;

    LAB_DEFAULT_CONSTRUCT(CircleCollider2DComponent);
};

struct RigidBodyComponent
{
    EBodyType body_type = EBodyType::Static;
    ECollisionDetectionType collision_detection = ECollisionDetectionType::Discrete;
    double mass = 1.0;
    double linear_drag = 0.01;
    double angular_drag = 0.05;
    bool disable_gravity = false;
    bool active = false;
    math::Vector3d initial_linear_velocity;
    math::Vector3d initial_angular_velocity;
    double max_linear_velocity = 500.0;
    double max_angular_velocity = 50.0;

    LAB_DEFAULT_CONSTRUCT(RigidBodyComponent);
};

}

}
