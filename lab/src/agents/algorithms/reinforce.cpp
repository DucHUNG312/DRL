#include "lab/agents/algorithms/reinforce.h"
#include "lab/agents/body.h"
#include "lab/utils/policy.h"
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
    auto action = utils::sample_action_with_policy(spec_. action_policy, shared_from_this(), state);
    return torch::IValue();
}

// sample state list in experience list
torch::Tensor Reinforce::sample()
{
    auto states = body_->memory()->sample().at("state");
    std::vector<torch::Tensor> tensor_vector;
    tensor_vector.reserve(states.size());
    for (const auto& state : states.vec())
        tensor_vector.push_back(state.toTensor());
    return torch::stack(tensor_vector).to(net_->device());
}

torch::Tensor Reinforce::calc_pdparam(torch::Tensor x)
{
    return net_->forward(x);
}

}

}