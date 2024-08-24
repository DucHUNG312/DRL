#pragma once

#include "lab/core.h"
#include "lab/utils/utils.h"

namespace lab
{
namespace spaces
{

class Space
{
public:
    using Iterator = torch::OrderedDict<std::string, std::shared_ptr<Space>>::Iterator;
    using ConstIterator = torch::OrderedDict<std::string, std::shared_ptr<Space>>::ConstIterator;

    torch::OrderedDict<std::string, torch::Tensor> parameters_;
    torch::OrderedDict<std::string, std::shared_ptr<Space>> children_;
    torch::Tensor shape_;
    std::string name_ = "Space";
    utils::Rand rand_;
public:
    explicit Space(std::string name);
    LAB_DEFAULT_CONSTRUCT(Space);

    virtual std::shared_ptr<Space> clone(const std::optional<torch::Device>& device = std::nullopt) const;
    
    void save(torch::serialize::OutputArchive& archive) const;

    void load(torch::serialize::InputArchive& archive);

    template <typename SpaceType>
    SpaceType* as() noexcept;

    template <typename SpaceType>
    const SpaceType* as() const noexcept;

    template <typename SpaceType>
    std::shared_ptr<SpaceType> register_subspace(std::string name, std::shared_ptr<SpaceType> space);

    template <typename SpaceType>
    std::shared_ptr<SpaceType> register_subspace(std::string name, utils::SpaceHolder<SpaceType> space_holder);

    torch::Tensor& register_parameter(std::string name, torch::Tensor tensor, bool requires_grad = false);
    
    torch::Tensor& get_parameter(std::string name);

    bool is_serializable() const;

    virtual void pretty_print(std::ostream& stream) const;

    void pretty_print_recursive(std::ostream& stream, const std::string& indentation) const;

    std::ostream& operator<<(std::ostream& stream);

    Iterator begin() 
    {
        return children_.begin();
    }

    ConstIterator begin() const 
    {
        return children_.begin();
    }

    Iterator end() 
    {
        return children_.end();
    }

    ConstIterator end() const 
    {
        return children_.end();
    }
private:
    template <typename Derived>
    friend class ClonableSpace;

    virtual void clone_(Space& other, const std::optional<torch::Device>& device);
};

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Space>& space);

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Space>& space);

template <typename SpaceType>
LAB_FORCE_INLINE SpaceType* Space::as() noexcept
{
    return dynamic_cast<SpaceType*>(this);
}

template <typename SpaceType>
LAB_FORCE_INLINE const SpaceType* Space::as() const noexcept
{
    return dynamic_cast<const SpaceType*>(this);
}

template <typename SpaceType>
LAB_FORCE_INLINE std::shared_ptr<SpaceType> Space::register_subspace(std::string name, std::shared_ptr<SpaceType> space)
{
    LAB_CHECK(!name.empty());
    LAB_CHECK(name.find('.') == std::string::npos);
    auto& base_space = children_.insert(std::move(name), std::move(space));
    return std::dynamic_pointer_cast<SpaceType>(base_space);
}

template <typename SpaceType>
LAB_FORCE_INLINE std::shared_ptr<SpaceType> Space::register_subspace(std::string name, utils::SpaceHolder<SpaceType> space_holder)
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
  class Name : public lab::utils::SpaceHolder<ImplType> { /* NOLINT */          \
   public:                                                                      \
    using lab::utils::SpaceHolder<ImplType>::SpaceHolder;                       \
  }

#define LAB_SPACE(Name) LAB_SPACE_IMPL(Name, Name##Impl)


}
}