#pragma once

#include "lab/common/common.h"

namespace lab 
{
namespace utils 
{

struct Placeholder 
{
    explicit Placeholder(const std::type_info& type_info_) noexcept
        : type_info(type_info_) {}
    Placeholder(const Placeholder&) = default;
    Placeholder(Placeholder&&) = default;
    virtual ~Placeholder() = default;
    virtual std::unique_ptr<Placeholder> clone() const 
    {
        TORCH_CHECK(false, "clone() should only be called on `AnyValue::Holder`");
    }
    const std::type_info& type_info;
};

}

}