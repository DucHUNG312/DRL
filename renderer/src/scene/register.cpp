#include "renderer/scene/register.h"
#include "renderer/utils/uuid.h"

namespace lab
{

namespace renderer
{

std::unordered_map<std::string, entt::entity>& Registry::get_entities()
{
    return entities_;
}

entt::entity Registry::create_entity(const std::optional<std::string>& name /*= std::nullopt*/)
{
    entt::entity entity = registry_.create();
    std::string uuid = UUID::gen();
    add_component<IDComponent>(entity, uuid);
    entities_.insert({ uuid, entity });
    if (name.has_value())
        add_component<NameComponent>(entity, name.value());
    return entity;
}

std::string Registry::create(const std::optional<std::string>& name /*= std::nullopt*/)
{
    entt::entity entity = create_entity(name);
    return get_id(entity);
}

entt::entity Registry::create_child(entt::entity parent, const std::optional<std::string>& name /*= std::nullopt*/)
{
    entt::entity child = create_entity(name);
    set_child(parent, child);
    set_parent(child, parent);
    return child;
}

entt::entity Registry::create_child(const std::string parent_id, const std::optional<std::string>& name /*= std::nullopt*/)
{
    return create_child(get_by_id(parent_id), name);
}

entt::entity Registry::create_parent(entt::entity child, const std::optional<std::string>& name /*= std::nullopt*/)
{
    entt::entity parent = create_entity(name);
    set_child(parent, child);
    set_parent(child, parent);
    return parent;
}

entt::entity Registry::create_parent(const std::string child_id, const std::optional<std::string>& name /*= std::nullopt*/)
{
    return create_parent(get_by_id(child_id), name);
}

entt::entity Registry::register_entity(const std::string& id, entt::entity entity)
{
    LAB_CHECK(entities_.find(id) == entities_.end());
    registry_.emplace<IDComponent>(entity, id);
    auto iter = entities_.insert({ id, entity }).first;
    return iter->second;
}

void Registry::unregister_entity(entt::entity entity)
{
    std::string id = get_id(entity);
    unregister_entity(id);
}

void Registry::unregister_entity(const std::string& id)
{
    LAB_CHECK(entities_.find(id) != entities_.end());
    registry_.destroy(entities_[id]);
    entities_.erase(id);
}

void Registry::clear()
{
    std::unordered_map<std::string, entt::entity>().swap(entities_);
    registry_.clear();
}

bool Registry::entity_exist(const std::string& id)
{
    if(entities_.find(id) == entities_.end())
        return false;
    return true;
}

entt::entity Registry::get_by_id(const std::string& id)
{
    if(entities_.find(id) != entities_.end())
        return entities_[id];
    LAB_LOG_WARN("There is no entity associated with this ID!");
    return entt::null;
}

std::vector<entt::entity> Registry::get_by_name(const std::string& name)
{
    std::vector<entt::entity> entities;
    for(auto entity: get_entities_with_components<NameComponent>())
        if(get_name(entity) == name)
            entities.push_back(entity);
    return entities; 
}

std::string Registry::get_id(entt::entity entity)
{
    assert(is_entity_valid(entity));
    return registry_.get<IDComponent>(entity).id;
}

std::string Registry::get_name(entt::entity entity)
{
    auto component = try_get_components<NameComponent>(entity);
    if(component != nullptr)
        return component->name;
    
    LAB_LOG_WARN("entity does not have a Name Component!");
    return "";
}

std::string Registry::get_name(const std::string& id)
{
    return get_name(get_by_id(id));
}

entt::entity Registry::get_parent(entt::entity entity)
{
    auto component = try_get_components<ParentComponent>(entity);
    if(component != nullptr)
        return component->parent;
    LAB_LOG_WARN("entity does not have a Parent Component!");
    return entt::null;
}

entt::entity Registry::get_parent(const std::string& id)
{
    return get_parent(get_by_id(id));
}

std::list<entt::entity>& Registry::get_children(entt::entity entity)
{
    auto component = try_get_components<ChildrenComponent>(entity);
    if(component != nullptr)
        return component->children;
    throw std::runtime_error("entity does not have a Children Component!");
}

std::list<entt::entity>& Registry::get_children(const std::string& id)
{
    return get_children(get_by_id(id));
}

size_t Registry::delete_name(entt::entity entity)
{
    return delete_components<NameComponent>(entity);
}

size_t Registry::delete_name(const std::string& id)
{
    return delete_name(get_by_id(id));
}

size_t Registry::delete_parent(entt::entity entity)
{
    return delete_components<ParentComponent>(entity);
}

size_t Registry::delete_parent(const std::string& id)
{
    return delete_parent(get_by_id(id));
}

size_t Registry::delete_children(entt::entity entity)
{

    return delete_components<ChildrenComponent>(entity);
}

size_t Registry::delete_children(const std::string& id)
{
    return delete_children(get_by_id(id));
}

void Registry::delete_child(entt::entity entity, entt::entity child)
{
    auto& children = get_children(entity);
    auto it = std::find_if(children.begin(), children.end(), [&child](entt::entity child_) {
        return child_ == child;
    });
    if (it != children.end()) children.erase(it);
    else LAB_LOG_WARN("entity is not a child!");
}

void Registry::delete_child(const std::string& id, entt::entity child)
{
    delete_child(get_by_id(id), child);
}

void Registry::set_name(entt::entity entity, const std::string& new_name)
{
    auto component = try_get_components<NameComponent>(entity);
    
    if (component != nullptr)
        component->name = new_name;
    else
    {
        NameComponent new_component;
        new_component.name = new_name;
        add_component<NameComponent>(entity, std::move(new_component));
    }
}

void Registry::set_name(const std::string& id, const std::string& new_name)
{
    set_name(get_by_id(id), new_name);
}

void Registry::set_parent(entt::entity entity, entt::entity new_parent)
{
    auto component = try_get_components<ParentComponent>(entity);
    
    if (component != nullptr)
    {
        if(component->parent == entt::null) component->parent = new_parent;
        else if(get_id(component->parent) != get_id(new_parent))
        {
            delete_child(component->parent, entity);
            component->parent = new_parent;
        }
        else return; // already is parent
    }
    else
    {
        ParentComponent new_component;
        new_component.parent = new_parent;
        add_component<ParentComponent>(entity, std::move(new_component));
    }
    //set_child(new_parent, entity); // should not implicit set new child ?
}

void Registry::set_parent(const std::string& id, entt::entity new_parent)
{
    set_parent(get_by_id(id), new_parent);
}

void Registry::set_child(entt::entity entity, entt::entity new_child)
{
    auto component = try_get_components<ChildrenComponent>(entity);
    
    if (component != nullptr)
    {
        auto it = std::find_if(component->children.begin(), component->children.end(), [&new_child](entt::entity child) {
            return child == new_child;
        });

        if (it == component->children.end()) component->children.push_back(new_child);
        else return; // already is children
    }
    else
    {
        ChildrenComponent new_component;
        new_component.children.push_back(new_child);
        add_component<ChildrenComponent>(entity, std::move(new_component));
    }
    //set_parent(new_child, entity); // should not implicit set new parent ?
}

void Registry::set_child(const std::string& id, entt::entity new_child)
{
    set_child(get_by_id(id), new_child);
}

bool Registry::is_entity_valid(entt::entity entity) const 
{
    return registry_.valid(entity);
}

bool Registry::is_entity_valid(const std::string& id)
{
    return is_entity_valid(get_by_id(id));
}

}

}