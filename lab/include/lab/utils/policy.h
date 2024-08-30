#pragma once

#include "lab/core.h"
#include "lab/utils/spec.h"
#include "lab/utils/typetraits.h"
#include "lab/distributions/bernoulli.h"
#include "lab/distributions/beta.h"
#include "lab/distributions/categorical.h"
#include "lab/distributions/cauchy.h"
#include "lab/distributions/dirichlet.h"
#include "lab/distributions/normal.h"

LAB_TYPE_DECLARE(Bernoulli, lab::distributions);
LAB_TYPE_DECLARE(Beta, lab::distributions);
LAB_TYPE_DECLARE(Categorical, lab::distributions);
LAB_TYPE_DECLARE(Cauchy, lab::distributions);
LAB_TYPE_DECLARE(Dirichlet, lab::distributions);
LAB_TYPE_DECLARE(Normal, lab::distributions);

namespace lab
{
namespace agents
{
class Algorithm;
}

namespace utils
{
struct VarScheduler
{
    LAB_ARG(VarSchedulerSpec, spec);
public:
    VarScheduler(const VarSchedulerSpec& spec);
    LAB_DEFAULT_CONSTRUCT(VarScheduler);
    double update();
};

struct NoDecay
{
    static constexpr const char* name = "no_decay";
    static double update(const VarSchedulerSpec& exp_var, int64_t step);
};
struct LinearDecay
{
    static constexpr const char* name = "linear_decay";
    static double update(const VarSchedulerSpec& exp_var, int64_t step);
};

struct RateDecay
{
    static constexpr const char* name = "rate_decay";
    static double update(const VarSchedulerSpec& exp_var, int64_t step, double decay_rate = 0.9, int64_t frequency = 20);
};

struct PeriodicDecay
{
    static constexpr const char* name = "periodic_decay";
    static double update(const VarSchedulerSpec& exp_var, int64_t step, int64_t frequency = 60);
};

struct ActionPolicy
{
    LAB_DEFAULT_CONSTRUCT(ActionPolicy);
};

struct DefaultPolicy : public ActionPolicy
{
    static constexpr const char* name = "default";
    static torch::Tensor sample(const std::shared_ptr<agents::Algorithm>& algorithm, const torch::Tensor& state); 
};

struct RandomPolicy : public ActionPolicy
{
    static constexpr const char* name = "random";
    static torch::Tensor sample(const std::shared_ptr<agents::Algorithm>& algorithm, const torch::Tensor& state);
};

struct EpsilonGreedyPolicy : public ActionPolicy
{
    static constexpr const char* name = "epsilon_greedy";
    static torch::Tensor sample(const std::shared_ptr<agents::Algorithm>& algorithm, const torch::Tensor& state); 
};

struct BoltzmannPolicy : public ActionPolicy
{
    static constexpr const char* name = "boltzmann";
    static torch::Tensor sample(const std::shared_ptr<agents::Algorithm>& algorithm, const torch::Tensor& state);
};

using Updaters = types_t<NoDecay, LinearDecay, RateDecay, PeriodicDecay>;
using DiscreteActionPDs = types_t<Bernoulli, Categorical>;
using ContinuousActionPDs = types_t<Beta, Cauchy, Dirichlet, Normal>;
using ActionPolicies = types_t<DefaultPolicy, RandomPolicy, EpsilonGreedyPolicy, BoltzmannPolicy>;

constexpr named_factory_t<double, update_call_maker, Updaters> UpdaterCallFactory;
constexpr named_factory_t<std::shared_ptr<distributions::Distribution>, shared_ptr_maker, DiscreteActionPDs> DiscreteActionPDFactory;
constexpr named_factory_t<std::shared_ptr<distributions::Distribution>, shared_ptr_maker, ContinuousActionPDs> ContinuousActionPDFactory;
constexpr named_factory_t<ActionPolicy, object_maker, ActionPolicies> ActionPolicyFactory;
constexpr named_factory_t<torch::Tensor, sample_call_maker, ActionPolicies> ActionPolicySampleFactory;

std::shared_ptr<distributions::Distribution> init_action_pd(std::string_view name, const torch::Tensor& pdparam);

torch::Tensor calc_pdparam(torch::Tensor state, const std::shared_ptr<agents::Algorithm>& algorithm);

torch::Tensor sample_action(std::string_view pdname, const torch::Tensor& pdparam);

ActionPolicy create_action_policy(std::string_view policy_name);

}

}