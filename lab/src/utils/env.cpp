#include "lab/utils/env.h"
#include "lab/utils/spec.h"
#include "lab/envs/base.h"
#include "lab/envs/classic_control/cartpole.h"

namespace lab
{
namespace utils
{

std::shared_ptr<envs::Env> create_env(const EnvSpec& spec)
{
    return EnvFactory(spec.name, spec);
}

StepResult::StepResult(
    const torch::Tensor& state,
    const torch::Tensor& next_state,
    const torch::Tensor& action,
    double reward, 
    bool terminated, 
    bool truncated)
    : state(state.to(torch::kDouble)),
    next_state(next_state.to(torch::kDouble)),
    action(action.to(torch::kDouble)),
    reward(reward), 
    terminated(terminated), 
    truncated(truncated) {}

StepResult StepResult::clone() const
{
    StepResult result;
    result.state = state.clone();
    result.next_state = next_state.clone();
    result.action = action.clone();
    result.reward = reward;
    result.terminated = terminated;
    result.truncated = truncated;
    return result;
}

StepResult& StepResult::to(torch::Device device)
{
    state.to(device);
    next_state.to(device);
    action.to(device);
    return *this;
}

void StepResult::pretty_print(std::ostream& stream, const std::string& indentation) const
{
    //const std::string next_indentation = indentation + "  ";
    //stream << indentation << "Result" << "(\n";
    //stream << next_indentation << "State: " << state << "\n";
    // stream << next_indentation << "Action: " << action << "\n";
    // stream << next_indentation << "Reward: " << reward << "\n";
    // stream << next_indentation << "Terminated: " << (terminated ? "true" : "false") << "\n";
    // stream << next_indentation << "Truncated: " << (truncated ? "true" : "false") << "\n";
    //stream << next_indentation << "Next State: " << next_state << "\n";
    //stream << indentation << ")";

    stream << indentation << "state: " << state << "\n";
    stream << indentation << "action: " << action << "; ";
    stream << indentation << "reward: " << reward << "; ";
    stream << indentation << "terminated: " << (terminated ? "true" : "false") << "; ";
    stream << indentation << "truncated: " << (truncated ? "true" : "false") << "\n";
}

std::ostream& operator<<(std::ostream& stream, const StepResult& result)
{
    result.pretty_print(stream, "");
    return stream;
}

}
}
