#pragma once

#include "lab/core.h"
#include "lab/utils/utils.h"

namespace lab
{
namespace spaces
{
enum SpaceType
{
    NONE,
    DISCRETE,
    BOX,
    TUPLE
};

template<typename T>
class SpaceBase : public torch::CustomClassHolder
{
    LAB_ARG(SpaceType, name) = SpaceType::NONE;
    LAB_ARG(utils::IShape, shape);
    LAB_ARG(utils::Rand, rand);
public:
    using Type = T;

    SpaceBase() = default;
    virtual ~SpaceBase() = default;

    SpaceBase(const SpaceBase& other)
        : name_(other.name_),
          rand_(other.rand_),
          shape_(other.shape_)
    {
    }

    SpaceBase(SpaceBase&& other) noexcept
        : name_(std::move(other.name_)),
          rand_(std::move(other.rand_)),
          shape_(std::move(other.shape_))
    {
    }

    SpaceBase& operator=(const SpaceBase& other) 
    {
        if (this != &other) 
        {
            name_ = other.name_;
            rand_ = other.rand_;
            shape_ = other.shape_;
        }
        return *this;
    }

    SpaceBase& operator=(SpaceBase&& other) noexcept 
    {
        if (this != &other) {
            name_ = std::move(other.name_);
            rand_ = std::move(other.rand_);
            shape_ = std::move(other.shape_);
        }
        return *this;
    }

    void set_seed(int64_t seed = 0)
    {
        rand_.set_seed(seed);
    }

    virtual T sample() = 0;

    virtual bool contains(const T& x) const = 0;
};

}
}