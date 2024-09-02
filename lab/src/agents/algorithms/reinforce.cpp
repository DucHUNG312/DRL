#include "lab/agents/algorithms/reinforce.h"
#include "lab/utils/policy.h"
#include "lab/utils/tensor.h"
#include "lab/utils/convert.h"
#include "lab/utils/math.h"

namespace lab
{

namespace agents
{

torch::Tensor Reinforce::train(const Algorithm::ExperienceDict& experiences)
{
    torch::Tensor states = calc_pdparam_batch(experiences);
    torch::Tensor advs = calc_ret_advs(experiences);
    torch::Tensor loss = calc_policy_loss(experiences, states, advs);
    net_->train_step(loss, optimizer_, lrscheduler_, env_->clock());
    to_train_ = false;
    return loss;
}

void Reinforce::update()
{
    explore_var_ = explore_var_scheduler_.update(env_->clock()->frame);
    entropy_coef_ = entropy_coef_scheduler_.update(env_->clock()->frame);
}

torch::Tensor Reinforce::act(const torch::Tensor& state)
{
    torch::Tensor action = utils::sample_action_with_policy(spec_. action_policy, shared_from_this(), state);
    return action.squeeze().to(torch::kCPU);
}

torch::Tensor Reinforce::calc_ret_advs(const Algorithm::ExperienceDict& experiences)
{
    auto reward_batch = experiences.at("reward");
    auto done_batch = experiences.at("terminated");
    auto rewards = utils::get_rewards_from_ivalue_list(reward_batch);
    auto dones = utils::get_dones_from_ivalue_list(done_batch);
    auto rets = utils::calc_returns(rewards, dones, spec_.gamma);
    torch::Tensor advs = torch::tensor(rets);
    if(center_return_)
        advs = utils::center_mean(advs.clone());
    if (env_->is_venv())
        advs = utils::venv_unpack(advs.clone());
    return advs.to(torch::kDouble);
}

torch::Tensor Reinforce::calc_policy_loss(const Algorithm::ExperienceDict& experiences, const torch::Tensor& states, const torch::Tensor& advs)
{
    std::shared_ptr<distributions::Distribution> action_pd = utils::init_action_pd(spec_.action_pdtype, states.clone().to(torch::kCPU));
    auto batch = experiences.at("action");
    torch::Tensor actions = utils::get_tensor_from_ivalue_list(batch);
    if(env_->is_venv())
        actions = utils::venv_unpack(actions.clone());
    torch::Tensor log_probs = action_pd->log_prob(actions.clone());
    torch::Tensor policy_loss = - spec_.policy_loss_coef * (log_probs * advs).mean();
    torch::Tensor entropy = action_pd->entropy().mean();
    policy_loss = policy_loss + (-entropy_coef_ * entropy);
    return policy_loss;
}


}

}