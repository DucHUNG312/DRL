#pragma once

#include "lab/core.h"
#include "lab/utils/typetraits.h"

namespace lab
{

namespace utils
{

template <typename Contained>
class SpaceHolder : public SpaceHolderIndicator
{
protected:
    /// The pointer this class wraps.
    std::shared_ptr<Contained> impl_;

public:
    using ContainedType = Contained;

    SpaceHolder() : impl_(default_construct()) 
    {
        static_assert(std::is_default_constructible<Contained>::value);
    }

    template <typename Head, typename... Tail, typename = typename std::enable_if<
          !(is_space_holder_of<Head, ContainedType>::value &&
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

}

}