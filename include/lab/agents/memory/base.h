#pragma once

#include "lab/agents/base.h"
#include "lab/utils/spec_types.h"

namespace lab
{
namespace agents
{

template<typename ObsSpace, typename ActSpace>
class Body;

template<typename ObsSpace, typename ActSpace>
class Memory
{
public:
    using DataKeyDict = torch::Dict<std::string, torch::IValue>;
    using ObsType = typename ObsSpace::Type;
    using ActType = typename ActSpace::Type;
    using BodyType = Body<ObsSpace, ActSpace>;

    LAB_ARG(utils::MemorySpec, spec);
    LAB_ARG(BodyType, body);
public:
    Memory() = default;
    Memory(const utils::MemorySpec& spec, const BodyType& body)
        : spec_(spec), body_(body) {}
    virtual ~Memory() = default;
    Memory(const Memory& other)
        : spec_(other.spec_), body_(other.body_) {}
    Memory(Memory&& other) noexcept
        : spec_(std::move(other.spec_)), body_(std::move(other.body_)) {}
    Memory& operator=(const Memory& other) 
    {
        if (this != &other) 
        {
            spec_ = other.spec_;
            body_ = other.body_;
        }
        return *this;
    }
    Memory& operator=(Memory&& other) noexcept 
    {
        if (this != &other) {
            spec_ = std::move(other.spec_);
            body_ = std::move(other.body_);
        }
        return *this;
    }

    virtual void reset() = 0;
    virtual void update() = 0;
    virtual void sample() = 0;
};

}
}