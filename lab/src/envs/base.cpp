#include "lab/spaces/base.h"
#include "lab/spaces/discrete.h"
#include "lab/spaces/box.h"
#include "lab/envs/base.h"

namespace lab
{
namespace envs
{

StepResult::StepResult(
    const torch::Tensor& state, 
    const torch::Tensor& action,
    double reward, 
    bool terminated, 
    bool truncated)
    : state(state.to(torch::kDouble)),
    next_state(torch::empty_like(state, torch::TensorOptions().dtype(torch::kDouble))),
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

Env::Env(const utils::EnvSpec& env_spec)
        : renderer::Scene(env_spec.name), 
          env_spec_(env_spec),
          rand_(env_spec.seed),
          clock_(std::make_shared<utils::Clock>(env_spec.max_frame, env_spec.clock_speed))
{}

bool Env::done() const
{
    return result_.terminated;
}

bool Env::is_venv() const
{
    return (env_spec_.num_envs > 1);
}

void Env::enable_rendering()
{
    env_spec_.renderer.enabled = true;
    render::Renderer::init();
    render();
}

const StepResult& Env::get_result() const
{
    return result_;
}

const utils::EnvSpec& Env::get_env_spec() const
{
    return env_spec_;
}

std::shared_ptr<spaces::AnySpace>& Env::get_observation_spaces()
{
    return observation_spaces_;
}

std::shared_ptr<spaces::AnySpace>& Env::get_action_spaces()
{
    return action_spaces_;
}

int64_t Env::get_state_dim() const
{
    LAB_CHECK(result_.state.defined());
    return result_.state.numel();
}

int64_t Env::get_action_dim() const
{
    return action_spaces_->ptr()->dim();
}

void Env::save(torch::serialize::OutputArchive& archive) const
{
    observation_spaces_->ptr()->save(archive);
    action_spaces_->ptr()->save(archive);

    archive.write("state", result_.state);
    archive.write("reward", result_.reward);
    archive.write("terminated", result_.terminated);
    archive.write("truncated", result_.truncated);
    archive.write("action", result_.action);
    archive.write("next_state", result_.next_state);
}

void Env::load(torch::serialize::InputArchive& archive)
{
    observation_spaces_->ptr()->load(archive);
    action_spaces_->ptr()->load(archive);

    torch::IValue reward;
    torch::IValue terminated;
    torch::IValue truncated;

    archive.read("state", result_.state);
    archive.read("next_state", result_.next_state);
    archive.read("action", result_.action);
    archive.read("reward", reward);
    archive.read("terminated", terminated);
    archive.read("truncated", truncated);

    result_.reward = reward.toDouble();
    result_.terminated = terminated.toBool();
    result_.truncated = truncated.toBool();
}

bool Env::is_serializable() const 
{ 
    return true; 
//  return false; 
}

void Env::pretty_print(std::ostream& stream, const std::string& indentation) const
{
    stream << env_spec_.name;
    stream << "(\n";
    const std::string next_indentation = indentation + "  ";

    stream << next_indentation << "Observation Space" << "(\n";
    observation_spaces_->ptr()->pretty_print_recursive(stream, next_indentation);
    stream << next_indentation << ")\n";
    stream << next_indentation << "Action Space" << "(\n";
    action_spaces_->ptr()->pretty_print_recursive(stream, next_indentation);
    stream << next_indentation << ")\n";
    result_.pretty_print(stream, next_indentation);

    stream << "\n";
    stream << indentation << ")";
}

std::ostream& operator<<(std::ostream& stream, const Env& env)
{
    env.pretty_print(stream, "");
    return stream;
}

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Env>& env)
{
    LAB_CHECK(env != nullptr);
    env->save(archive);
    return archive;
}

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Env>& env)
{
    LAB_CHECK(env != nullptr);
    env->load(archive);
    return archive;
}

}
}