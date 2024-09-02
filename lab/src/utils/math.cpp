#include "lab/utils/math.h"

namespace lab
{
namespace utils
{

std::vector<double> calc_returns(const std::vector<double>& rewards, const std::vector<bool>& dones, double gamma)
{
    LAB_CHECK(rewards.size() == dones.size());
    std::vector<double> rets(rewards.size(), 0);
    double future_ret = 0;
    std::vector<bool> not_dones(dones.size());
    std::transform(dones.begin(), dones.end(), not_dones.begin(), [](bool done){
        return !done;
    });

    for (int64_t t = rewards.size() - 1; t >= 0; --t) 
    {
        future_ret = rewards[t] + gamma * future_ret * not_dones[t];
        rets[t] = future_ret;
    }
    return rets;
}

std::vector<double> calc_nstep_returns(const std::vector<double>& rewards, const std::vector<bool>& dones, double next_v_pred, double gamma)
{
    LAB_CHECK(rewards.size() == dones.size());
    std::vector<double> rets(rewards.size(), 0);
    double future_ret = next_v_pred;
    std::vector<bool> not_dones;
    std::transform(dones.begin(), dones.end(), std::back_inserter(not_dones), [](bool done){
        return !done;
    });

    for (int64_t t = rewards.size() - 1; t >= 0; --t) 
    {
        future_ret = rewards[t] + gamma * future_ret * not_dones[t];
        rets[t] = future_ret;
    }
    return rets;
}

std::vector<double> calc_q_value_logits(const std::vector<double>& rewards, const std::vector<bool>& dones, double next_v_pred, double gamma)
{
    LAB_CHECK(rewards.size() == dones.size());
    std::vector<double> rets(rewards.size(), 0);
    double future_ret = next_v_pred;
    std::vector<bool> not_dones;
    std::transform(dones.begin(), dones.end(), std::back_inserter(not_dones), [](bool done){
        return !done;
    });

    for (int64_t t = rewards.size() - 1; t >= 0; --t) 
    {
        future_ret = rewards[t] + gamma * future_ret * not_dones[t];
        rets[t] = future_ret;
    }
    return rets;
}



}

}