#pragma once

#include "lab/core.h"
#include "lab/utils/policy.h"

namespace lab
{

namespace utils
{

std::vector<double> calc_returns(const std::vector<double>& rewards, const std::vector<bool>& dones, double gamma);

std::vector<double> calc_nstep_returns(const std::vector<double>& rewards, const std::vector<bool>& dones, double next_v_pred, double gamma);

std::vector<double> calc_q_value_logits(const std::vector<double>& rewards, const std::vector<bool>& dones, double next_v_pred, double gamma);

double no_decay(const VarScheduler& exp_var, int64_t step);

double linear_decay(const VarScheduler& exp_var, int64_t step);

double rate_decay(const VarScheduler& exp_var, int64_t step, double decay_rate = 0, int64_t frequency = 20);

double periodic_decay(const VarScheduler& exp_var, int64_t step, int64_t frequency = 20);

}

}
