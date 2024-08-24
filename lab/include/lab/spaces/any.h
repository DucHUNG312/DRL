#pragma once

#include "lab/core.h"
#include "lab/utils/anyspaceholder.h"

namespace lab
{

namespace spaces
{

class AnySpace
{
public:
    AnySpace() = default;
    AnySpace(AnySpace&&) = default;
    AnySpace& operator=(AnySpace&&) = default;

    template <typename SpaceType>
    AnySpace(std::shared_ptr<SpaceType> space)
        : content_(make_holder(std::move(space), &std::remove_reference<SpaceType>::type::forward)) 
    {
        static_assert(utils::is_space<SpaceType>::value, "Can only store object derived from Space into AnySpace");
        static_assert(utils::has_sample_and_contains<SpaceType>::value);
    }

    template <typename SpaceType, typename T>
    AnySpace(SpaceType&& space)
        : AnySpace(std::make_shared<SpaceType>(std::forward<SpaceType>(space))) {}

    template <typename SpaceType>
    AnySpace(const utils::SpaceHolder<SpaceType>& space_holder)
        : AnySpace(space_holder.ptr()) {}

    inline AnySpace(const AnySpace& other)
        : content_(other.content_ ? other.content_->copy() : nullptr) {}

    inline AnySpace& operator=(const AnySpace& other) 
    {
        if (this != &other) 
            content_ = other.content_ ? other.content_->copy() : nullptr;
        return *this;
    }

    inline AnySpace clone(std::optional<torch::Device> device) const 
    {
        AnySpace clone;
        clone.content_ = content_ ? content_->clone_space(device) : nullptr;
        return clone;
    }

    template <typename SpaceType>
    AnySpace& operator=(std::shared_ptr<SpaceType> space) 
    {
        // NOLINTNEXTLINE(cppcoreguidelines-c-copy-assignment-signature)
        return (*this = AnySpace(std::move(space)));
    }

    torch::nn::AnyValue any_sample(torch::Tensor&& mask) 
    {
        LAB_CHECK(!is_empty());
        return content_->sample(std::move(mask));
    }

    template <typename ReturnType>
    ReturnType sample(torch::Tensor&& mask) 
    {
        return any_sample(std::forward<torch::Tensor>(mask)).template get<ReturnType>();
    }

    template<typename ArgumentType>
    bool contains(ArgumentType&& argument)
    {
        LAB_CHECK(!is_empty());
        return content_->contains(std::move(torch::nn::AnyValue(std::move(argument))));
    }

    template <typename T, typename>
    T& get() 
    {
        LAB_CHECK(!is_empty());
        return get_<T>();
    }

    template <typename T, typename>
    const T& get() const
    {
        LAB_CHECK(!is_empty());
        return get_<T>();
    }

    template <typename T, typename ContainedType>
    T get() const 
    {
        return T(ptr<ContainedType>());
    }

    inline std::shared_ptr<spaces::Space> ptr() const 
    {
        LAB_CHECK(!is_empty());
        return content_->ptr();
    }

    template <typename T, typename>
    std::shared_ptr<T> ptr() const 
    {
        LAB_CHECK(!is_empty());
        get_<T>();
        return std::dynamic_pointer_cast<T>(ptr());
    }

    inline const std::type_info& type_info() const 
    {
        LAB_CHECK(!is_empty());
        return content_->type_info;
    }

    inline bool is_empty() const noexcept 
    {
        return content_ == nullptr;
    }
private:
    template <typename SpaceType, typename Class, typename ReturnType, typename ArgumentType>
    std::unique_ptr<utils::AnySpacePlaceholder> make_holder(std::shared_ptr<SpaceType>&& space, ReturnType (Class::*)(ArgumentType)) 
    {
        static_assert(torch::detail::check_not_lvalue_references<ArgumentType>(),
            "spaces stored inside AnySpace must not take references. "
            "Use pointers instead.");
        static_assert(
            !std::is_void<ReturnType>::value,
            "AnySpace cannot store spaces that return void "
            "(you can return a dummy value).");
        return std::make_unique<utils::AnySpaceHolder<std::decay_t<SpaceType>, ArgumentType>>(std::move(space)); 
    }

    template <typename SpaceType>
    SpaceType& get_() const 
    {
        using S = typename std::remove_reference<SpaceType>::type;
        static_assert(utils::has_sample_and_contains<S>::value, "Can only call AnySpace::get<T> with a type T that has a sample and contains method");
        return get_(&S::forward);
    }

    template <typename SpaceType, typename ReturnType, typename ArgumentType>
    SpaceType& get_(ReturnType (SpaceType::*)(ArgumentType)) const 
    {
        if (typeid(SpaceType).hash_code() == type_info().hash_code()) 
            return *static_cast<utils::AnySpaceHolder<SpaceType, ArgumentType>&>(*content_).space;
        LAB_LOG_FATAL("Attempted to cast space of type {} to type {}", c10::demangle(type_info().name()), c10::demangle(typeid(SpaceType).name()));
    }
        /// The type erased Space.
    std::unique_ptr<utils::AnySpacePlaceholder> content_;

};


class NamedAnySpace 
{
public:
    template <typename SpaceType>
    NamedAnySpace(std::string name, std::shared_ptr<SpaceType> space_ptr)
        : NamedAnySpace(std::move(name), AnySpace(std::move(space_ptr))) {}

    template <typename M, typename = utils::enable_if_space_t<M>>
    NamedAnySpace(std::string name, M&& space)
        : NamedAnySpace(std::move(name), std::make_shared<typename std::remove_reference<M>::type>(std::forward<M>(space))) {}

    template <typename M>
    NamedAnySpace(std::string name, const utils::SpaceHolder<M>& space_holder)
        : NamedAnySpace(std::move(name), space_holder.ptr()) {}

    NamedAnySpace(std::string name, AnySpace any_space)
        : name_(std::move(name)), space_(std::move(any_space)) {}

    const std::string& name() const noexcept 
    {
        return name_;
    }

    AnySpace& space() noexcept 
    {
        return space_;
    }

    const AnySpace& space() const noexcept 
    {
        return space_;
    }

    private:
    std::string name_;
    AnySpace space_;
};




}

}