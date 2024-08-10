#pragma once

#include "lab/core.h"
#include "lab/utils/rand.h"

namespace lab
{
namespace spaces
{
enum SpaceType
{
    NONE,
    DISCRETE,
    BOX
};

template<typename T>
class SpaceBase
{
public:
    SpaceBase(int seed = 0)
        : seed(seed)
    {}

    ~SpaceBase() = default;

    virtual T sample() = 0;

    virtual bool contains(T x) = 0;
public:
    int seed;

    SpaceType name = SpaceType::NONE; 

    torch::Generator generator;

    torch::IntArrayRef shape;
};
}
}