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
    std::vector<bool> not_dones;
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

double no_decay(const VarScheduler& exp_var, int64_t step)
{
    return exp_var.spec().start_val;
}

double linear_decay(const VarScheduler& exp_var, int64_t step)
{
    if(step < exp_var.spec().start_step)
        return exp_var.spec().start_val;
    double slope = (exp_var.spec().end_val - exp_var.spec().start_val) / (exp_var.spec().end_step - exp_var.spec().start_step);
    double val = std::max(slope * (step - exp_var.spec().start_step) + exp_var.spec().start_val, exp_var.spec().end_val);
    return val;
}

double rate_decay(const VarScheduler& exp_var, int64_t step, double decay_rate /*= 0*/, int64_t frequency /*= 20*/)
{
    if(step < exp_var.spec().start_step)
        return exp_var.spec().start_val;
    if(step >= exp_var.spec().end_step)
        return exp_var.spec().end_val;
    int64_t step_per_decay = (exp_var.spec().end_step - exp_var.spec().start_step) / frequency;
    int64_t decay_step = (step - exp_var.spec().start_step) / step_per_decay;
    double val = std::max(std::pow(decay_rate, decay_step) * exp_var.spec().start_val, exp_var.spec().end_val);
    return val;

}

double periodic_decay(const VarScheduler& exp_var, int64_t step, int64_t frequency /*= 20*/)
{
    if(step < exp_var.spec().start_step)
        return exp_var.spec().start_val;
    if(step >= exp_var.spec().end_step)
        return exp_var.spec().end_val;
    int64_t x_freq = frequency;
    int64_t step_per_decay = (exp_var.spec().end_step - exp_var.spec().start_step) / x_freq;
    int64_t x = (step - exp_var.spec().start_step) / step_per_decay;
    double unit = exp_var.spec().start_val - exp_var.spec().end_val;
    double val = exp_var.spec().end_val * 0.5 * unit * (1 + std::cos(x) * (1 - x / x_freq));
    return std::max(val, exp_var.spec().end_val);
}

}

}