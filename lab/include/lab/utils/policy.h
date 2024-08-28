#pragma once

#include "lab/core.h"
#include "lab/utils/spec.h"
#include "lab/utils/net.h"
#include "lab/utils/math.h"
#include "lab/utils/rand.h"
#include "lab/utils/typetraits.h"
#include "lab/distributions/base.h"

namespace lab
{

namespace utils
{

double no_decay(const VarSchedulerSpec& exp_var, int64_t step);

double linear_decay(const VarSchedulerSpec& exp_var, int64_t step);

double rate_decay(const VarSchedulerSpec& exp_var, int64_t step, double decay_rate = 0, int64_t frequency = 20);

double periodic_decay(const VarSchedulerSpec& exp_var, int64_t step, int64_t frequency = 20);

struct VarScheduler
{
    LAB_ARG(VarSchedulerSpec, spec);
public:
    LAB_DEFAULT_CONSTRUCT(VarScheduler);
    VarScheduler(const VarSchedulerSpec& spec)
        : spec_(spec) {}

    double update()
    {
        int64_t step = 0; // TODO
        switch (spec_.updater)
        {
            case VarUpdater::NO_DECAY: return no_decay(spec_, step);
            case VarUpdater::LINEAR_DECAY: return linear_decay(spec_, step);
            case VarUpdater::RATE_DECAY: return rate_decay(spec_, step);
            case VarUpdater::PERIODIC_DECAY: return periodic_decay(spec_, step);
            default: LAB_UNREACHABLE;
        }
        return 0;
    }
};

template<typename T, typename = enable_if_algorithm_t<T>>
LAB_FORCE_INLINE torch::Tensor calc_pdparam (torch::Tensor state, const std::shared_ptr<T>& algorithm)
{
    return algorithm->calc_pdparam(state.to(get_torch_device()));
}

template<typename Distribution, typename = void>
LAB_FORCE_INLINE Distribution init_action_pd(const torch::Tensor& pdparam)
{
    LAB_UNIMPLEMENTED;
    throw std::runtime_error("Unimplemented");
};

template<typename Distribution>
LAB_FORCE_INLINE lab::utils::enable_if_discrete_pd_t<Distribution> init_action_pd(const torch::Tensor& pdparam)
{
    return Distribution(pdparam, true);
}

template<typename Distribution>
LAB_FORCE_INLINE lab::utils::enable_if_continuous_pd_t<Distribution> init_action_pd(const torch::Tensor& pdparam)
{
    torch::Tensor loc_scale = pdparam.transpose(0, 1);
    torch::Tensor loc = loc_scale[0];
    torch::Tensor scale = torch::clamp(loc_scale[1], -20, 2).exp();
    return Distribution(loc, scale);
}

template<typename Distribution>
LAB_FORCE_INLINE torch::Tensor sample_action(const torch::Tensor& pdparam)
{
    Distribution action_pd = init_action_pd<Distribution>(pdparam);
    return action_pd.sample();
}

template<typename Distribution, typename Body, typename Algorithm, ActionPolicyType>
struct ActionPolicy
{
    static torch::Tensor sample(const torch::Tensor& state, const std::shared_ptr<Algorithm>& algorithm)
    {
        return torch::Tensor();
    }
};

template<typename Distribution, typename Body, typename Algorithm>
struct ActionPolicy<Distribution, Body, Algorithm, ActionPolicyType::ACTION_POLICY_NONE>
{
    static torch::Tensor sample(const torch::Tensor& state, const std::shared_ptr<Algorithm>& algorithm)
    {
        torch::Tensor pdparam = calc_pdparam(state, algorithm);
        return sample_action<Distribution>(pdparam);
    }
};

template<typename Distribution, typename Body, typename Algorithm>
struct ActionPolicy<Distribution, Body, Algorithm, ActionPolicyType::RANDOM_POLICY>
{
    static torch::Tensor sample(const torch::Tensor& state, const std::shared_ptr<Algorithm>& algorithm)
    { 
        return Body::get_action_space()->sample();
    }
};

template<typename Distribution, typename Body, typename Algorithm>
struct ActionPolicy<Distribution, Body, Algorithm, ActionPolicyType::EPSILON_GREEDY>
{
    static torch::Tensor sample(const torch::Tensor& state, const std::shared_ptr<Algorithm>& algorithm)
    {
        if constexpr(Body::get_spec().explore_spec.start_val > Rand::rand())
            return ActionPolicy<Distribution, Body, Algorithm, ActionPolicyType::RANDOM_POLICY>::sample(state, algorithm);
        else
            return ActionPolicy<Distribution, Body, Algorithm, ActionPolicyType::ACTION_POLICY_NONE>::sample(state, algorithm);
    }
};

template<typename Distribution, typename Body, typename Algorithm>
struct ActionPolicy<Distribution, Body, Algorithm, ActionPolicyType::BOLTZMANN>
{
    static torch::Tensor sample(const torch::Tensor& state, const std::shared_ptr<Algorithm>& algorithm)
    {
       double tau = Body::get_spec().explore_spec.start_val;
       torch::Tensor pdparam = calc_pdparam<Algorithm>(state, algorithm) / tau;
       return sample_action<Distribution>(pdparam);
    }
};

}

}