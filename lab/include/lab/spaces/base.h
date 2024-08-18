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
    SpaceBase(const SpaceBase& other) = default;
    SpaceBase(SpaceBase&& other) noexcept = default;
    SpaceBase& operator=(const SpaceBase& other) = default;
    SpaceBase& operator=(SpaceBase&& other) noexcept = default;
    virtual ~SpaceBase() = default;

    void set_seed(int64_t seed = 0)
    {
        rand_.set_seed(seed);
    }

    virtual T sample() = 0;

    virtual bool contains(const T& x) const = 0;
};

}
}