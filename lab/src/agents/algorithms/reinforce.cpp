#include "lab/agents/algorithms/reinforce.h"

namespace lab
{

namespace agents
{

torch::Tensor Reinforce::train()
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

void Reinforce::update()
{
    LAB_UNIMPLEMENTED;
}

torch::IValue Reinforce::act(const torch::Tensor& state)
{
    LAB_UNIMPLEMENTED;
    return torch::IValue();
}

torch::Tensor Reinforce::sample()
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

torch::Tensor Reinforce::calc_pdparam(torch::Tensor x)
{
    return net_->forward(x);
}

}

}