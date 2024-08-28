#pragma once

#include "lab/agents/algorithms/base.h"
#include "lab/envs/base.h"

namespace lab
{

namespace agents
{

class Reinforce : public Algorithm
{
public:
    using Algorithm::Algorithm;

    torch::Tensor train();

    void update();

    torch::IValue act(const torch::Tensor& state);

    torch::Tensor sample();

    torch::Tensor calc_pdparam(torch::Tensor x);
private:
    bool to_train = false;
};

}

}