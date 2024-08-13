#pragma once

#include "lab/core.h"

namespace lab
{
namespace utils
{

class GlobalAdam : public torch::optim::Adam
{
public:
    explicit GlobalAdam( 
        std::vector<torch::optim::OptimizerParamGroup> param_groups,
        torch::optim::AdamOptions defaults = {})
        : Adam(
            std::move(param_groups),
            std::make_unique<torch::optim::AdamOptions>(defaults)) 
    {
        
    }

    explicit GlobalAdam(std::vector<torch::Tensor> params, torch::optim::AdamOptions defaults = {})
        : Adam({OptimizerParamGroup(std::move(params))}, defaults) 
    {

    }

    torch::Tensor step(torch::optim::LossClosure closure = nullptr) override
    {

    }
};
}
}