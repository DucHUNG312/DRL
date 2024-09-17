#pragma once

#include "lab/common/common.h"
#include "lab/utils/spec.h"

namespace lab {

namespace utils {
struct VarScheduler {
  LAB_ARG(VarSchedulerSpec, spec);

 public:
  VarScheduler(const VarSchedulerSpec& spec);
  LAB_DEFAULT_CONSTRUCT(VarScheduler);
  double update(int64_t step);
};

struct NoDecay {
  static constexpr const char* name = "no_decay";
  static double update(const VarSchedulerSpec& exp_var, int64_t step, double decay_rate = 0, int64_t frequency = 0);
};
struct LinearDecay {
  static constexpr const char* name = "linear_decay";
  static double update(const VarSchedulerSpec& exp_var, int64_t step, double decay_rate = 0, int64_t frequency = 0);
};

struct RateDecay {
  static constexpr const char* name = "rate_decay";
  static double update(const VarSchedulerSpec& exp_var, int64_t step, double decay_rate = 0.9, int64_t frequency = 20);
};

struct PeriodicDecay {
  static constexpr const char* name = "periodic_decay";
  static double update(const VarSchedulerSpec& exp_var, int64_t step, double decay_rate = 0, int64_t frequency = 60);
};

struct ActionPolicy {
  LAB_DEFAULT_CONSTRUCT(ActionPolicy);
};

struct DefaultPolicy : public ActionPolicy {
  static constexpr const char* name = "default";
  static torch::Tensor sample(const std::shared_ptr<agents::Algorithm>& algorithm, const torch::Tensor& state);
};

struct RandomPolicy : public ActionPolicy {
  static constexpr const char* name = "random";
  static torch::Tensor sample(const std::shared_ptr<agents::Algorithm>& algorithm, const torch::Tensor& state);
};

struct EpsilonGreedyPolicy : public ActionPolicy {
  static constexpr const char* name = "epsilon_greedy";
  static torch::Tensor sample(const std::shared_ptr<agents::Algorithm>& algorithm, const torch::Tensor& state);
};

struct BoltzmannPolicy : public ActionPolicy {
  static constexpr const char* name = "boltzmann";
  static torch::Tensor sample(const std::shared_ptr<agents::Algorithm>& algorithm, const torch::Tensor& state);
};

std::shared_ptr<distributions::Distribution> init_action_pd(std::string_view name, const torch::Tensor& pdparam);

torch::Tensor calc_pdparam(const std::shared_ptr<agents::Algorithm>& algorithm, torch::Tensor state);

torch::Tensor sample_action_with_pd(std::string_view pdname, const torch::Tensor& pdparam);

torch::Tensor sample_action_with_policy(
    std::string_view policy_name,
    const std::shared_ptr<agents::Algorithm>& algorithm,
    const torch::Tensor& state);

ActionPolicy create_action_policy(std::string_view policy_name);

} // namespace utils

} // namespace lab