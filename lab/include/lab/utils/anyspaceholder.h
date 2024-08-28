#pragma once

#include "lab/core.h"
#include "lab/utils/placeholder.h"

namespace lab 
{
namespace spaces 
{
class Space;
}
}

namespace lab 
{
namespace utils 
{

struct AnySpacePlaceholder : public Placeholder
{
    using Placeholder::Placeholder;
    
    virtual torch::nn::AnyValue sample(torch::Tensor&& arguments) = 0;

    virtual bool contains(torch::nn::AnyValue&& arguments) = 0;

    /// Returns std::shared_ptr<Space> pointing to the erased space.
    virtual std::shared_ptr<spaces::Space> ptr() = 0;

    /// Returns a `AnySpacePlaceholder` with a shallow copy of this `AnySpace`.
    virtual std::unique_ptr<AnySpacePlaceholder> copy() const = 0;

    /// Returns a `AnySpacePlaceholder` with a deep copy of this `AnySpace`.
    virtual std::unique_ptr<AnySpacePlaceholder> clone_space(std::optional<torch::Device> device) const = 0;
};

template <typename SpaceType, typename ArgumentTypes>
struct AnySpaceHolder : public AnySpacePlaceholder 
{
    struct CheckedGetter 
    {
        template <typename T>
        std::decay_t<T>&& operator()() 
        {
            if (auto* maybe_value = arguments_.template try_get<std::decay_t<T>>()) 
            {
                return std::move(*maybe_value);
            }
            LAB_LOG_FATAL("Expected argument to be of type {}, but received value of type {}", c10::demangle(typeid(T).name()), c10::demangle(arguments_.type_info().name()));
        }
        torch::nn::AnyValue arguments_;
    };

    struct InvokeSample
    {
        torch::nn::AnyValue operator()(torch::Tensor&& ts) 
        {
            return torch::nn::AnyValue(space_->sample(std::forward<torch::Tensor>(ts)));
        }
        std::shared_ptr<SpaceType>& space_;
    };

    struct InvokeContains
    {
        template <typename Ts>
        bool operator()(Ts&& ts) 
        {
            return space_->contains(std::forward<Ts>(ts));
        }
        std::shared_ptr<SpaceType>& space_;
    };

    explicit AnySpaceHolder(std::shared_ptr<SpaceType>&& space_)
        : AnySpacePlaceholder(typeid(SpaceType)), space(std::move(space_)) {}

    
    torch::nn::AnyValue sample(torch::Tensor&& argument) override 
    {    
        return torch::unpack<torch::nn::AnyValue, torch::Tensor>(InvokeSample{space}, CheckedGetter{torch::nn::AnyValue(std::move(argument))});
    }

    bool contains(torch::nn::AnyValue&& argument) override 
    {    
        return torch::unpack<bool, ArgumentTypes>(InvokeContains{space}, CheckedGetter{argument});
    }

    std::shared_ptr<spaces::Space> ptr() override 
    {
        return space;
    }

    std::unique_ptr<AnySpacePlaceholder> copy() const override 
    {
        return std::make_unique<AnySpaceHolder>(*this);
    }

    std::unique_ptr<AnySpacePlaceholder> clone_space(std::optional<torch::Device> device) const override 
    {
        return std::make_unique<AnySpaceHolder>(std::dynamic_pointer_cast<SpaceType>(space->clone(device)));
    }

    /// The actual concrete space instance.
    std::shared_ptr<SpaceType> space;
};

}
}
