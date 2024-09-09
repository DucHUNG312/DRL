#pragma once

#include "lab/common/common.h"
#include "lab/utils/typetraits.h"

namespace lab
{

namespace utils
{

using Envs = types_t<envs::CartPole>;
constexpr named_factory_t<std::shared_ptr<envs::Env>, shared_ptr_maker, Envs> EnvFactory;

std::shared_ptr<envs::Env> create_env(const EnvSpec& spec);

struct StepResult
{
    torch::Tensor state;
    torch::Tensor next_state;
    torch::Tensor action;
    double reward = 0;
    bool terminated = false; 
    bool truncated = false;

    StepResult(
        const torch::Tensor& state, 
        const torch::Tensor& next_state,
        const torch::Tensor& action,
        double reward, 
        bool terminated, 
        bool truncated);
    LAB_DEFAULT_CONSTRUCT(StepResult);

    StepResult clone() const;

    StepResult& to(torch::Device device); 

    void pretty_print(std::ostream& stream, const std::string& indentation) const;
};

std::ostream& operator<<(std::ostream& stream, const StepResult& result);

}
}