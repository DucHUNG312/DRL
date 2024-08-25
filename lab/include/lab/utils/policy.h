#pragma once

#include "lab/core.h"
#include "lab/utils/spec.h"

namespace lab
{

namespace utils
{

struct VarScheduler
{
    LAB_ARG(ExploreVarSpec, spec);
public:
    LAB_DEFAULT_CONSTRUCT(VarScheduler);
    VarScheduler(const ExploreVarSpec& spec)
        : spec_(std::move(spec)) {}

    double update()
    {
        return 0;
    }
};

}

}