#include "lab/utils/policy.h"
#include "lab/agents/algorithms/base.h"
#include "lab/distributions/bernoulli.h"
#include "lab/distributions/beta.h"
#include "lab/distributions/categorical.h"
#include "lab/distributions/cauchy.h"
#include "lab/distributions/dirichlet.h"
#include "lab/distributions/normal.h"
#include "lab/utils/rand.h"
#include "lab/utils/typetraits.h"

namespace lab {
namespace utils {

LAB_TYPE_DECLARE(Bernoulli, lab::distributions);
LAB_TYPE_DECLARE(Beta, lab::distributions);
LAB_TYPE_DECLARE(Categorical, lab::distributions);
LAB_TYPE_DECLARE(Cauchy, lab::distributions);
LAB_TYPE_DECLARE(Dirichlet, lab::distributions);
LAB_TYPE_DECLARE(Normal, lab::distributions);

using Updaters = types_t<NoDecay, LinearDecay, RateDecay, PeriodicDecay>;
using DiscreteActionPDs = types_t<Bernoulli, Categorical>;
using ContinuousActionPDs = types_t<Beta, Cauchy, Dirichlet, Normal>;
using ActionPolicies = types_t<DefaultPolicy, RandomPolicy, EpsilonGreedyPolicy, BoltzmannPolicy>;

constexpr named_factory_t<double, update_call_maker, Updaters> UpdaterCallFactory;
constexpr named_factory_t<std::shared_ptr<distributions::Distribution>, shared_ptr_maker, DiscreteActionPDs>
    DiscreteActionPDFactory;
constexpr named_factory_t<std::shared_ptr<distributions::Distribution>, shared_ptr_maker, ContinuousActionPDs>
    ContinuousActionPDFactory;
constexpr named_factory_t<ActionPolicy, object_maker, ActionPolicies> ActionPolicyFactory;
constexpr named_factory_t<torch::Tensor, sample_call_maker, ActionPolicies> ActionPolicySampleFactory;

VarScheduler::VarScheduler(const VarSchedulerSpec& spec) : spec_(spec) {}

double VarScheduler::update(int64_t step) {
  return UpdaterCallFactory(spec_.updater, spec_, step);
}

double NoDecay::update(
    const VarSchedulerSpec& exp_var,
    int64_t step,
    double decay_rate /*= 0*/,
    int64_t frequency /*= 0*/) {
  return exp_var.start_val;
}

double LinearDecay::update(
    const VarSchedulerSpec& exp_var,
    int64_t step,
    double decay_rate /*= 0*/,
    int64_t frequency /*= 0*/) {
  if (step < exp_var.start_step)
    return exp_var.start_val;
  double slope = (exp_var.end_val - exp_var.start_val) / (exp_var.end_step - exp_var.start_step);
  double val = std::max(slope * (step - exp_var.start_step) + exp_var.start_val, exp_var.end_val);
  return val;
}

double RateDecay::update(
    const VarSchedulerSpec& exp_var,
    int64_t step,
    double decay_rate /*= 0.9*/,
    int64_t frequency /*= 20*/) {
  if (step < exp_var.start_step)
    return exp_var.start_val;
  if (step >= exp_var.end_step)
    return exp_var.end_val;
  int64_t step_per_decay = (exp_var.end_step - exp_var.start_step) / frequency;
  int64_t decay_step = (step - exp_var.start_step) / step_per_decay;
  double val = std::max(std::pow(decay_rate, decay_step) * exp_var.start_val, exp_var.end_val);
  return val;
}

double PeriodicDecay::update(
    const VarSchedulerSpec& exp_var,
    int64_t step,
    double decay_rate /*= 0*/,
    int64_t frequency /*= 60*/) {
  if (step < exp_var.start_step)
    return exp_var.start_val;
  if (step >= exp_var.end_step)
    return exp_var.end_val;
  int64_t x_freq = frequency;
  int64_t step_per_decay = (exp_var.end_step - exp_var.start_step) / x_freq;
  int64_t x = (step - exp_var.start_step) / step_per_decay;
  double unit = exp_var.start_val - exp_var.end_val;
  double val = exp_var.end_val * 0.5 * unit * (1 + std::cos(x) * (1 - x / x_freq));
  return std::max(val, exp_var.end_val);
}

torch::Tensor DefaultPolicy::sample(const std::shared_ptr<agents::Algorithm>& algorithm, const torch::Tensor& state) {
  torch::Tensor pdparam = calc_pdparam(algorithm, state);
  return sample_action_with_pd(algorithm->spec().action_pdtype, pdparam);
}

torch::Tensor RandomPolicy::sample(const std::shared_ptr<agents::Algorithm>& algorithm, const torch::Tensor& state) {
  return algorithm->env()->get_action_spaces()->sample();
}

torch::Tensor EpsilonGreedyPolicy::sample(
    const std::shared_ptr<agents::Algorithm>& algorithm,
    const torch::Tensor& state) {
  if (algorithm->spec().explore_spec.start_val > Rand::rand())
    return DefaultPolicy::sample(algorithm, state);

  return RandomPolicy::sample(algorithm, state);
}

torch::Tensor BoltzmannPolicy::sample(const std::shared_ptr<agents::Algorithm>& algorithm, const torch::Tensor& state) {
  double tau = algorithm->spec().explore_spec.start_val;
  torch::Tensor pdparam = calc_pdparam(algorithm, state) / tau;
  return sample_action_with_pd(algorithm->spec().action_pdtype, pdparam);
}

torch::Tensor calc_pdparam(const std::shared_ptr<agents::Algorithm>& algorithm, torch::Tensor state) {
  return algorithm->calc_pdparam(state.to(get_torch_device()).to(torch::kDouble));
}

std::shared_ptr<distributions::Distribution> init_action_pd(std::string_view name, const torch::Tensor& pdparam) {
  if (name == "Bernoulli" || name == "Categorical")
    return DiscreteActionPDFactory(name, pdparam, true);

  torch::Tensor loc_scale = pdparam.transpose(0, 1);
  torch::Tensor loc = loc_scale[0];
  torch::Tensor scale = torch::clamp(loc_scale[1], -20, 2).exp();
  return ContinuousActionPDFactory(name, loc, scale);
}

torch::Tensor sample_action_with_pd(std::string_view pdname, const torch::Tensor& pdparam) {
  auto action_pd = init_action_pd(pdname, pdparam);
  return action_pd->sample();
}

torch::Tensor sample_action_with_policy(
    std::string_view policy_name,
    const std::shared_ptr<agents::Algorithm>& algorithm,
    const torch::Tensor& state) {
  return ActionPolicySampleFactory(policy_name, algorithm, state);
}

ActionPolicy create_action_policy(std::string_view policy_name) {
  return ActionPolicyFactory(policy_name);
}

} // namespace utils

} // namespace lab