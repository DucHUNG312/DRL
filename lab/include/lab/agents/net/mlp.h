#pragma once

#include "lab/agents/net/base.h"

namespace lab
{

namespace agents
{

class MLPNetImpl : public NetImpl
{
    LAB_ARG(torch::nn::Sequential, model);
    LAB_ARG(torch::nn::ModuleList, model_tail);
public:
    static constexpr const char* name = "MLPNet";
public:
    MLPNetImpl(const utils::NetSpec& spec, int64_t in_dim, torch::IntArrayRef out_dim);

    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(MLPNet);

}

}