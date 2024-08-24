#pragma once

#include "renderer/core.h"
#include "renderer/scene/components.h"

namespace lab
{

namespace renderer
{

class Registry
{
public:
    Registry() = default;
    Registry(Registry&& other) noexcept = default;
    Registry& operator=(Registry&& other) noexcept = default;
    virtual ~Registry() = default;
    LAB_NONCOPYABLE(Registry);

    std::unordered_map<std::string, entt::entity>& get_entities();

    entt::entity create_entity(const std::optional<std::string>& name = std::nullopt);

    std::string create(const std::optional<std::string>& name = std::nullopt);

    entt::entity create_child(entt::entity parent, const std::optional<std::string>& name = std::nullopt);

    entt::entity create_child(const std::string parent_id, const std::optional<std::string>& name = std::nullopt);

    entt::entity create_parent(entt::entity child, const std::optional<std::string>& name = std::nullopt);

    entt::entity create_parent(const std::string child_id, const std::optional<std::string>& name = std::nullopt);

    entt::entity register_entity(const std::string& id, entt::entity entity);

    void unregister_entity(entt::entity entity);

    void unregister_entity(const std::string& id);

    void clear();

    bool entity_exist(const std::string& id);

    entt::entity get_by_id(const std::string& id);

    std::vector<entt::entity> get_by_name(const std::string& name);

    std::string get_id(entt::entity entity);

    std::string get_name(entt::entity entity);

    std::string get_name(const std::string& id);

    entt::entity get_parent(entt::entity entity);

    entt::entity get_parent(const std::string& id);

    std::list<entt::entity>& get_children(entt::entity entity);

    std::list<entt::entity>& get_children(const std::string& id);

    size_t delete_name(entt::entity entity);

    size_t delete_name(const std::string& id);

    size_t delete_parent(entt::entity entity);

    size_t delete_parent(const std::string& id);

    size_t delete_children(entt::entity entity);

    size_t delete_children(const std::string& id);

    void delete_child(entt::entity entity, entt::entity child);

    void delete_child(const std::string& id, entt::entity child);

    void set_name(entt::entity entity, const std::string& new_name);

    void set_name(const std::string& id, const std::string& new_name);

    void set_parent(entt::entity entity, entt::entity new_parent);

    void set_parent(const std::string& id, entt::entity new_parent);

    void set_child(entt::entity entity, entt::entity new_child);

    void set_child(const std::string& id, entt::entity new_child);

    bool is_entity_valid(entt::entity entity) const;

    bool is_entity_valid(const std::string& id);

    template<typename... Components>
    bool has_components(entt::entity entity);

    template<typename... Components>
    bool has_components(const std::string& id);

    template<typename Component, typename... Args>
    auto add_component(entt::entity entity, Args&&... args);

    template<typename Component, typename... Args>
    auto add_component(const std::string& id, Args&&... args);

    template <typename Component, typename... Args>
    auto get_or_add_component(entt::entity entity, Args&&... args);

    template <typename Component, typename... Args>
    auto get_or_add_component(const std::string& id, Args&&... args);

    template <typename Component, typename... Args>
    auto add_or_replace_component(entt::entity entity, Args&&... args);

    template <typename Component, typename... Args>
    auto add_or_replace_component(const std::string& id, Args&&... args);

    template<typename Component, typename... Args>
    void copy_component_to(entt::entity entity, entt::entity dst_entity, const Registry& dst_registry, Args&&... args);

    template<typename Component, typename... Args>
    void copy_component_to(const std::string& id, const std::string& dst_id, const Registry& dst_registry, Args&&... args);

    template <typename... Components>
    auto get_components(entt::entity entity);

    template <typename... Components>
    auto get_components(const std::string& id);

    template <typename... Components>
    auto try_get_components(entt::entity entity);

    template <typename... Components>
    auto try_get_components(const std::string& id);

    template <typename... Components>
    size_t delete_components(entt::entity entity);

    template<typename... Component>
    size_t delete_components(const std::string& id);

    template <typename... Components>
    size_t try_delete_component(entt::entity entity);

    template <typename... Components>
    size_t try_delete_component(const std::string& id);

    template<typename... Components>
    void delete_entities_with_components();

    template <typename R, typename T>
    void add_dependency();

    template<typename Component, typename... Args>
    void add_component_to_all(Args&&... args);

    template<typename... Components>
    auto delete_components_from_all();

    template<typename... Components>
    auto get_entities_with_components();
private:
    void copy_registry(const entt::registry &source, entt::registry &destination);
private:
    std::unordered_map<std::string, entt::entity> entities_;
    entt::registry registry_;
};

template<typename... Components>
LAB_FORCE_INLINE bool Registry::has_components(entt::entity entity)
{
    assert(is_entity_valid(entity));
    return registry_.all_of<Components...>(entity);
}

template<typename... Components>
LAB_FORCE_INLINE bool Registry::has_components(const std::string& id)
{
    return has_components<Components...>(get_by_id(id));
}

template<typename Component, typename... Args>
LAB_FORCE_INLINE auto Registry::add_component(entt::entity entity, Args&&... args)
{
    assert(is_entity_valid(entity));
    if(has_components<Component>(entity))
        LAB_LOG_WARN("Attempting to add a component twice");
    return registry_.emplace<Component>(entity, std::forward<Args>(args)...);
}

template<typename Component, typename... Args>
LAB_FORCE_INLINE auto Registry::add_component(const std::string& id, Args&&... args)
{
    return add_component<Component, Args...>(get_by_id(id), std::forward<Args>(args)...);
}

template <typename Component, typename... Args>
LAB_FORCE_INLINE auto Registry::get_or_add_component(entt::entity entity, Args&&... args)
{
    assert(is_entity_valid(entity));
    return registry_.get_or_emplace<Component>(entity, std::forward<Args>(args)...);
}

template <typename Component, typename... Args>
LAB_FORCE_INLINE auto Registry::get_or_add_component(const std::string& id, Args&&... args)
{
    return get_or_add_component<Component, Args...>(get_by_id(id), std::forward<Args>(args)...);
}

template <typename Component, typename... Args>
LAB_FORCE_INLINE auto Registry::add_or_replace_component(entt::entity entity, Args&&... args)
{
    assert(is_entity_valid(entity));
    return registry_.emplace_or_replace<Component>(entity, std::forward<Args>(args)...);
}

template <typename Component, typename... Args>
LAB_FORCE_INLINE auto Registry::add_or_replace_component(const std::string& id, Args&&... args)
{
    return add_or_replace_component<Component, Args...>(get_by_id(id), std::forward<Args>(args)...);
}

template<typename Component, typename... Args>
LAB_FORCE_INLINE void Registry::copy_component_to(entt::entity entity, entt::entity dst_entity, const Registry& dst_registry, Args&&... args)
{
    if(is_entity_valid(entity))
    {
        if(!has_components<Component>(entity)) LAB_LOG_FATAL("source entity does not have this component!");
        else dst_registry.template add_or_replace_component<Component, Args...>(dst_entity, std::forward<Args>(args)...);
    }
    else LAB_LOG_WARN("entity does not exists in this registry");
}

template<typename Component, typename... Args>
LAB_FORCE_INLINE void Registry::copy_component_to(const std::string& id, const std::string& dst_id, const Registry& dst_registry, Args&&... args)
{
    copy_component_to<Component, Args...>(get_by_id(id), get_by_id(dst_id), dst_registry, std::forward<Args>(args)...);
}

template <typename... Components>
LAB_FORCE_INLINE auto Registry::get_components(entt::entity entity)
{
    assert(is_entity_valid(entity));
    return registry_.get<Components...>(entity);
}

template <typename... Components>
LAB_FORCE_INLINE auto Registry::get_components(const std::string& id)
{
    return get_components<Components...>(get_by_id(id));
}

template <typename... Components>
LAB_FORCE_INLINE auto Registry::try_get_components(entt::entity entity)
{
    assert(is_entity_valid(entity));
    return registry_.try_get<Components...>(entity);
}

template <typename... Components>
LAB_FORCE_INLINE auto Registry::try_get_components(const std::string& id)
{
    return try_get_components<Components...>(get_by_id(id));
}

template <typename... Components>
LAB_FORCE_INLINE size_t Registry::delete_components(entt::entity entity)
{
    assert(is_entity_valid(entity));
    return registry_.remove<Components...>(entity);
}

template<typename... Component>
LAB_FORCE_INLINE size_t Registry::delete_components(const std::string& id)
{
    return delete_components<Component...>(get_by_id(id));
}

template <typename... Components>
LAB_FORCE_INLINE size_t Registry::try_delete_component(entt::entity entity)
{
    if(has_components<Components...>(entity))
        return delete_components<Components...>(entity);
    return 0;
}

template <typename... Components>
LAB_FORCE_INLINE size_t Registry::try_delete_component(const std::string& id)
{
    return try_delete_component<Components...>(get_by_id(id));
}

template<typename... Components>
LAB_FORCE_INLINE void Registry::delete_entities_with_components()
{
    auto view = registry_.view<Components...>();
    for(auto entity : view)
        unregister_entity(entity);
}

template <typename R, typename T>
LAB_FORCE_INLINE void Registry::add_dependency()
{
    registry_.on_construct<R>().template connect<&entt::registry::get_or_emplace<T>>();
}

template<typename Component, typename... Args>
LAB_FORCE_INLINE void Registry::add_component_to_all(Args&&... args)
{
    for (auto entity: registry_.view<entt::entity>())
        add_component<Component, Args...>(entity, std::forward<Args>(args)...);
}

template<typename... Components>
LAB_FORCE_INLINE auto Registry::delete_components_from_all()
{
    for (auto entity: registry_.view<entt::entity>())
        delete_components<Components...>(entity);
}

template<typename... Components>
LAB_FORCE_INLINE auto Registry::get_entities_with_components()
{
    return registry_.view<Components...>();
}

}

}

