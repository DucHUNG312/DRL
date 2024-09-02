#pragma once

#include "lab/core.h"
#include "lab/utils/rand.h"
#include "lab/utils/typetraits.h"

namespace lab
{
namespace spaces
{
    
template <typename Contained>
class SpaceHolder : public utils::SpaceHolderIndicator
{
protected:
    /// The pointer this class wraps around
    std::shared_ptr<Contained> impl_;
public:
    using ContainedType = Contained;

    SpaceHolder() : impl_(default_construct()) 
    {
        static_assert(std::is_default_constructible<Contained>::value);
    }

    template <typename Head, typename... Tail, typename = typename std::enable_if<
          !(utils::is_space_holder_of<Head, ContainedType>::value &&
            (sizeof...(Tail) == 0))>::type>
    explicit SpaceHolder(Head&& head, Tail&&... tail)
        : impl_(new Contained(std::forward<Head>(head), std::forward<Tail>(tail)...)) {}

    /* implicit */ SpaceHolder(std::nullptr_t) : impl_(nullptr) {}

    /* implicit */ SpaceHolder(std::shared_ptr<Contained> space)
        : impl_(std::move(space)) {}

    explicit operator bool() const noexcept 
    {
        return !is_empty();
    }

    Contained* operator->() 
    {
        return get();
    }

    const Contained* operator->() const 
    {
        return get();
    }

    Contained& operator*() 
    {
        return *get();
    }

    const Contained& operator*() const 
    {
        return *get();
    }

    const std::shared_ptr<Contained>& ptr() const 
    {
        LAB_CHECK(!is_empty());
        return impl_;
    }

    Contained* get() 
    {
        LAB_CHECK(!is_empty());
        return impl_.get();
    }

    const Contained* get() const 
    {
        LAB_CHECK(!is_empty());
        return impl_.get();
    }

    bool is_empty() const noexcept 
    {
        return impl_ == nullptr;
    }
private:
    template <typename T = Contained>
    std::shared_ptr<Contained> default_construct() 
    {
        if constexpr (std::is_default_constructible_v<T>) 
            return std::make_shared<Contained>();
        else
            return nullptr;
    }
};

class Space
{
public:
    using ParameterDict = torch::OrderedDict<std::string, torch::Tensor>;
    using ChildrenDict = torch::OrderedDict<std::string, std::shared_ptr<Space>>;
    using Iterator = ChildrenDict::Iterator;
    using ConstIterator = ChildrenDict::ConstIterator;

    LAB_ARG(ParameterDict, parameters);
    LAB_ARG(ChildrenDict, children);
    LAB_ARG(torch::Tensor, shape);
    LAB_ARG(std::string, name) = "Space";
    LAB_ARG(utils::Rand, rand);
    LAB_ARG(int64_t, dim) = -1;
public:
    explicit Space(std::string name);
    LAB_DEFAULT_CONSTRUCT(Space);

    virtual std::shared_ptr<Space> clone(const std::optional<torch::Device>& device = std::nullopt) const;
    
    void save(torch::serialize::OutputArchive& archive) const;

    void load(torch::serialize::InputArchive& archive);

    template <typename SpaceType>
    std::shared_ptr<SpaceType> register_subspace(std::string name, std::shared_ptr<SpaceType> space);

    template <typename SpaceType>
    std::shared_ptr<SpaceType> register_subspace(std::string name, SpaceHolder<SpaceType> space_holder);

    torch::Tensor& register_parameter(std::string name, torch::Tensor tensor, bool requires_grad = false);
    
    torch::Tensor& get_parameter(std::string name);

    bool is_serializable() const;

    virtual void pretty_print(std::ostream& stream) const;

    void pretty_print_recursive(std::ostream& stream, const std::string& indentation) const;

    std::ostream& operator<<(std::ostream& stream);

    Iterator begin();

    ConstIterator begin() const;

    Iterator end();

    ConstIterator end() const;

    template <typename SpaceType>
    typename SpaceType::ContainedType* as() noexcept;

    template <typename SpaceType>
    const typename SpaceType::ContainedType* as() const noexcept;

    template <typename SpaceType, typename = utils::disable_if_space_holder_t<SpaceType>>
    SpaceType* as() noexcept;

    template <typename SpaceType, typename = utils::disable_if_space_holder_t<SpaceType>>
    const SpaceType* as() const noexcept;
private:
    template <typename Derived>
    friend class ClonableSpace;

    virtual void clone_(Space& other, const std::optional<torch::Device>& device);
};

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Space>& space);

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Space>& space);

template <typename SpaceType>
typename SpaceType::ContainedType* Space::as() noexcept 
{
    return as<typename SpaceType::ContainedType>();
}

template <typename SpaceType>
const typename SpaceType::ContainedType* Space::as() const noexcept 
{
    return as<typename SpaceType::ContainedType>();
}

template <typename SpaceType, typename>
SpaceType* Space::as() noexcept 
{
    return dynamic_cast<SpaceType*>(this);
}

template <typename SpaceType, typename>
const SpaceType* Space::as() const noexcept 
{
    return dynamic_cast<const SpaceType*>(this);
}

template <typename SpaceType>
std::shared_ptr<SpaceType> Space::register_subspace(std::string name, std::shared_ptr<SpaceType> space)
{
    LAB_CHECK(!name.empty());
    LAB_CHECK(name.find('.') == std::string::npos);
    auto& base_space = children_.insert(std::move(name), std::move(space));
    return std::dynamic_pointer_cast<SpaceType>(base_space);
}

template <typename SpaceType>
std::shared_ptr<SpaceType> Space::register_subspace(std::string name, SpaceHolder<SpaceType> space_holder)
{
    return register_subspace(std::move(name), space_holder.ptr());
}

template<typename Derived>
class ClonableSpace : public Space
{
public:
    using Space::Space;

    // reset shape and rand inside Derived class
    virtual void reset() = 0;

    std::shared_ptr<Space> clone(const std::optional<torch::Device>& device = std::nullopt) const override 
    {
        torch::NoGradGuard no_grad;

        const auto& self = static_cast<const Derived&>(*this);
        auto copy = std::make_shared<Derived>(self);
        copy->parameters_.clear();
        copy->children_.clear();
        copy->reset();

        LAB_CHECK(copy->parameters_.size() == parameters_.size());
        for (const auto& parameter : parameters_) 
        {
            auto& tensor = *parameter;
            auto data = device && tensor.device() != *device ? tensor.to(*device) : torch::autograd::Variable(tensor).clone();
            copy->parameters_[parameter.key()].set_data(data);
        }
        
        LAB_CHECK(copy->children_.size() == children_.size());
        for (const auto& child : children_) 
        {
            copy->children_[child.key()]->clone_(*child.value(), device);
        }
        return copy;
    }
private:
    void clone_(Space& other, const std::optional<torch::Device>& device) final 
    {
        auto clone = std::dynamic_pointer_cast<Derived>(other.clone(device));
        LAB_CHECK(clone != nullptr);
        static_cast<Derived&>(*this) = *clone;
    }
};

#define LAB_SPACE_IMPL(Name, ImplType)                                          \
  class Name : public lab::spaces::SpaceHolder<ImplType> { /* NOLINT */          \
   public:                                                                      \
    using lab::spaces::SpaceHolder<ImplType>::SpaceHolder;                       \
  }

#define LAB_SPACE(Name) LAB_SPACE_IMPL(Name, Name##Impl)


}
}