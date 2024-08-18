#pragma once

#include "lab/core.h"

namespace lab
{
namespace utils
{
class NoGradGuard
{
public:
    NoGradGuard() 
    {
        no_grad_guard = std::make_unique<torch::NoGradGuard>();
    }

    virtual ~NoGradGuard() 
    {
        no_grad_guard.reset();
    }
private:
    std::unique_ptr<torch::NoGradGuard> no_grad_guard;
};
}
}