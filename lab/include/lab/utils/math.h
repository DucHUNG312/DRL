#pragma once

#include "lab/common/common.h"

namespace lab
{

namespace utils
{

std::vector<double> calc_returns(const std::vector<double>& rewards, const std::vector<bool>& dones, double gamma);

std::vector<double> calc_nstep_returns(const std::vector<double>& rewards, const std::vector<bool>& dones, double next_v_pred, double gamma);

std::vector<double> calc_q_value_logits(const std::vector<double>& rewards, const std::vector<bool>& dones, double next_v_pred, double gamma);

}

}
