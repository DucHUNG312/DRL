#include "lab/utils/policy.h"

namespace lab
{
namespace utils
{

double no_decay(const VarSchedulerSpec& exp_var, int64_t step)
{
    return exp_var.start_val;
}

double linear_decay(const VarSchedulerSpec& exp_var, int64_t step)
{
    if(step < exp_var.start_step)
        return exp_var.start_val;
    double slope = (exp_var.end_val - exp_var.start_val) / (exp_var.end_step - exp_var.start_step);
    double val = std::max(slope * (step - exp_var.start_step) + exp_var.start_val, exp_var.end_val);
    return val;
}

double rate_decay(const VarSchedulerSpec& exp_var, int64_t step, double decay_rate /*= 0*/, int64_t frequency /*= 20*/)
{
    if(step < exp_var.start_step)
        return exp_var.start_val;
    if(step >= exp_var.end_step)
        return exp_var.end_val;
    int64_t step_per_decay = (exp_var.end_step - exp_var.start_step) / frequency;
    int64_t decay_step = (step - exp_var.start_step) / step_per_decay;
    double val = std::max(std::pow(decay_rate, decay_step) * exp_var.start_val, exp_var.end_val);
    return val;

}

double periodic_decay(const VarSchedulerSpec& exp_var, int64_t step, int64_t frequency /*= 20*/)
{
    if(step < exp_var.start_step)
        return exp_var.start_val;
    if(step >= exp_var.end_step)
        return exp_var.end_val;
    int64_t x_freq = frequency;
    int64_t step_per_decay = (exp_var.end_step - exp_var.start_step) / x_freq;
    int64_t x = (step - exp_var.start_step) / step_per_decay;
    double unit = exp_var.start_val - exp_var.end_val;
    double val = exp_var.end_val * 0.5 * unit * (1 + std::cos(x) * (1 - x / x_freq));
    return std::max(val, exp_var.end_val);
}

}

}