#pragma once

#include "lab/core.h"
#include "lab/spaces/base.h"
#include "lab/utils/placeholder.h"

namespace lab
{

namespace spaces
{

struct AnySpacePlaceholder : public utils::Placeholder
{
    using utils::Placeholder::Placeholder;
    
    virtual torch::Tensor sample(/*torch::Tensor&& argument*/) = 0;

    virtual bool contains(torch::nn::AnyValue&& arguments) = 0;

    virtual std::shared_ptr<Space> ptr() = 0;

    virtual std::unique_ptr<AnySpacePlaceholder> copy() const = 0;

    virtual std::unique_ptr<AnySpacePlaceholder> clone_space(std::optional<torch::Device> device) const = 0;
};

template <typename SpaceType, typename... ArgumentTypes>
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
            AT_ERROR("Expected argument to be of type ", c10::demangle(typeid(T).name()), ", but received value of type ", c10::demangle(arguments_.type_info().name()));
        }
        
        torch::nn::AnyValue arguments_;
    };

    struct InvokeSample
    {
        torch::Tensor operator()(/*torch::Tensor&& ts*/) 
        {
            return torch::Tensor(space_->sample(/*std::forward<torch::Tensor>(ts)*/));
        }
        std::shared_ptr<SpaceType>& space_;
    };

    struct InvokeContains
    {
        template <typename... Ts>
        bool operator()(Ts&&... ts) 
        {
            return bool(space_->contains(std::forward<Ts>(ts)...));
        }
        std::shared_ptr<SpaceType>& space_;
    };

    explicit AnySpaceHolder(std::shared_ptr<SpaceType>&& space_)
        : AnySpacePlaceholder(typeid(SpaceType)), space(std::move(space_)) {}

    
    torch::Tensor sample(/*torch::Tensor&& argument*/) override 
    {    
        return torch::Tensor(InvokeSample{space}());
    }

    bool contains(torch::nn::AnyValue&& argument) override 
    {    
        return torch::unpack<bool, ArgumentTypes...>(InvokeContains{space}, CheckedGetter{argument});
    }

    std::shared_ptr<Space> ptr() override 
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

class AnySpace
{
public:
    AnySpace() = default;
    AnySpace(AnySpace&&) = default;
    AnySpace& operator=(AnySpace&&) = default;

    template <typename SpaceType>
    AnySpace(std::shared_ptr<SpaceType> space);

    template <typename SpaceType, typename T>
    AnySpace(SpaceType&& space);

    template <typename SpaceType>
    AnySpace(const SpaceHolder<SpaceType>& space_holder);

    AnySpace(const AnySpace& other);

    AnySpace& operator=(const AnySpace& other);

    AnySpace clone(std::optional<torch::Device> device) const;

    template <typename SpaceType>
    AnySpace& operator=(std::shared_ptr<SpaceType> space);

    torch::Tensor sample(/*torch::Tensor&& mask*/);

    template<typename ArgumentType>
    bool contains(ArgumentType&& argument);

    template <typename T, typename>
    T& get();

    template <typename T, typename>
    const T& get() const;

    template <typename T, typename ContainedType>
    T get() const;

    std::shared_ptr<Space> ptr() const;

    template <typename T, typename = utils::enable_if_space_t<typename T::ContainedType>>
    std::shared_ptr<typename T::ContainedType> ptr() const;

    const std::type_info& type_info() const;

    bool is_empty() const noexcept;
private:
    template <typename SpaceType, typename Class, typename ReturnType, typename... ArgumentType>
    std::unique_ptr<AnySpacePlaceholder> make_holder(std::shared_ptr<SpaceType>&& space, ReturnType (Class::*)(ArgumentType...));

    template <typename SpaceType>
    SpaceType& get_() const;

    template <typename SpaceType, typename ReturnType, typename... ArgumentType>
    SpaceType& get_(ReturnType (SpaceType::*)(ArgumentType...)) const;
private:
    std::unique_ptr<AnySpacePlaceholder> content_;
};

template <typename SpaceType>
AnySpace::AnySpace(std::shared_ptr<SpaceType> space)
    : content_(make_holder(std::move(space), &std::remove_reference<SpaceType>::type::sample)) 
{
    static_assert(utils::is_space<SpaceType>::value, "Can only store object derived from Space into AnySpace");
    static_assert(utils::has_sample_and_contains<SpaceType>::value);
}

template <typename SpaceType, typename T>
AnySpace::AnySpace(SpaceType&& space)
    : AnySpace(std::make_shared<SpaceType>(std::forward<SpaceType>(space))) {}

template <typename SpaceType>
AnySpace::AnySpace(const SpaceHolder<SpaceType>& space_holder)
    : AnySpace(space_holder.ptr()) {}

template <typename SpaceType>
AnySpace& AnySpace::operator=(std::shared_ptr<SpaceType> space) 
{
    // NOLINTNEXTLINE(cppcoreguidelines-c-copy-assignment-signature)
    return (*this = AnySpace(std::move(space)));
}

template<typename ArgumentType>
bool AnySpace::contains(ArgumentType&& argument)
{
    LAB_CHECK(!is_empty());
    return content_->contains(std::move(torch::nn::AnyValue(std::move(argument))));
}

template <typename T, typename>
T& AnySpace::get() 
{
    LAB_CHECK(!is_empty());
    return get_<T>();
}

template <typename T, typename>
const T& AnySpace::get() const
{
    LAB_CHECK(!is_empty());
    return get_<T>();
}

template <typename T, typename ContainedType>
T AnySpace::get() const 
{
    return T(ptr<ContainedType>());
}

template <typename T, typename>
std::shared_ptr<typename T::ContainedType> AnySpace::ptr() const 
{
    LAB_CHECK(!is_empty());
    get_<typename T::ContainedType>();
    return std::dynamic_pointer_cast<typename T::ContainedType>(ptr());
}

template <typename SpaceType, typename Class, typename ReturnType, typename... ArgumentType>
std::unique_ptr<AnySpacePlaceholder> AnySpace::make_holder(std::shared_ptr<SpaceType>&& space, ReturnType (Class::*)(ArgumentType...)) 
{
    static_assert(torch::detail::check_not_lvalue_references<ArgumentType...>(),
        "spaces stored inside AnySpace must not take references. "
        "Use pointers instead.");
    static_assert(
        !std::is_void<ReturnType>::value,
        "AnySpace cannot store spaces that return void "
        "(you can return a dummy value).");
    return std::make_unique<AnySpaceHolder<std::decay_t<SpaceType>, ArgumentType...>>(std::move(space)); 
}

template <typename SpaceType>
SpaceType& AnySpace::get_() const 
{
    using S = typename std::remove_reference<SpaceType>::type;
    static_assert(utils::has_sample_and_contains<S>::value, "Can only call AnySpace::get<T> with a type T that has a sample and contains method");
    return get_(&S::sample);
}

template <typename SpaceType, typename ReturnType, typename... ArgumentType>
SpaceType& AnySpace::get_(ReturnType (SpaceType::*)(ArgumentType...)) const 
{
    if (typeid(SpaceType).hash_code() == type_info().hash_code()) 
        return *static_cast<AnySpaceHolder<SpaceType, ArgumentType...>&>(*content_).space;
    AT_ERROR("Attempted to cast space of type ", c10::demangle(type_info().name()), " to type ", c10::demangle(typeid(SpaceType).name()));
}

class NamedAnySpace 
{
public:
    template <typename SpaceType>
    NamedAnySpace(std::string name, std::shared_ptr<SpaceType> space_ptr);

    template <typename M, typename = utils::enable_if_space_t<M>>
    NamedAnySpace(std::string name, M&& space);

    template <typename M>
    NamedAnySpace(std::string name, const SpaceHolder<M>& space_holder);

    NamedAnySpace(std::string name, AnySpace any_space);

    const std::string& name() const noexcept;

    AnySpace& space() noexcept;

    const AnySpace& space() const noexcept;
private:
    std::string name_;
    AnySpace space_;
};

template <typename SpaceType>
NamedAnySpace::NamedAnySpace(std::string name, std::shared_ptr<SpaceType> space_ptr)
    : NamedAnySpace(std::move(name), AnySpace(std::move(space_ptr))) {}

template <typename M, typename>
NamedAnySpace::NamedAnySpace(std::string name, M&& space)
    : NamedAnySpace(std::move(name), std::make_shared<typename std::remove_reference<M>::type>(std::forward<M>(space))) {}

template <typename M>
NamedAnySpace::NamedAnySpace(std::string name, const SpaceHolder<M>& space_holder)
    : NamedAnySpace(std::move(name), space_holder.ptr()) {}




}

}