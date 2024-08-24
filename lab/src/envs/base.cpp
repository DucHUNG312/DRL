#include "lab/envs/base.h"

namespace lab
{
namespace envs
{

StepResult::StepResult(const torch::Tensor& state, double reward, bool terminated, bool truncated)
    : state(state), reward(reward), terminated(terminated), truncated(truncated) {}

void StepResult::pretty_print(std::ostream& stream, const std::string& indentation) const
{
    const std::string next_indentation = indentation + "  ";
    stream << indentation << "Result" << "(\n";
    stream << next_indentation << "State: " << state << "\n";
    stream << next_indentation << "Reward: " << reward << "\n";
    stream << next_indentation << "Terminated: " << (terminated ? "true" : "false") << "\n";
    stream << next_indentation << "Truncated: " << (truncated ? "true" : "false") << "\n";
    stream << indentation << ")";
}

Env::Env(const utils::EnvSpec& env_spec)
        : renderer::Scene(env_spec.name), env_spec_(env_spec) 
{
}

void Env::reset(uint64_t seed /*= 0*/)
{
    LAB_UNIMPLEMENTED;
}

void Env::step(int64_t act)
{
    LAB_UNIMPLEMENTED;
}

int64_t Env::sample()
{
    LAB_UNIMPLEMENTED;
    return 0;
}

void Env::close()
{
    LAB_UNIMPLEMENTED;
}

void Env::render()
{
    LAB_UNIMPLEMENTED;
}

void Env::enable_rendering()
{
    LAB_UNIMPLEMENTED;
}

const StepResult& Env::get_result() const
{
    return result_;
}

const utils::EnvSpec& Env::get_env_spec() const
{
    return env_spec_;
}

std::shared_ptr<spaces::Space>& Env::get_observation_spaces()
{
    return observation_spaces_;
}

std::shared_ptr<spaces::Space>& Env::get_action_spaces()
{
    return action_spaces_;
}

void Env::save(torch::serialize::OutputArchive& archive) const
{
    observation_spaces_->save(archive);
    action_spaces_->save(archive);

    archive.write("state", result_.state);
    archive.write("reward", result_.reward);
    archive.write("terminated", result_.terminated);
    archive.write("truncated", result_.truncated);
}

void Env::load(torch::serialize::InputArchive& archive)
{
    observation_spaces_->load(archive);
    action_spaces_->load(archive);

    torch::IValue reward;
    torch::IValue terminated;
    torch::IValue truncated;

    archive.read("state", result_.state);
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
    observation_spaces_->pretty_print_recursive(stream, next_indentation);
    stream << next_indentation << ")\n";
    stream << next_indentation << "Action Space" << "(\n";
    action_spaces_->pretty_print_recursive(stream, next_indentation);
    stream << next_indentation << ")\n";
    result_.pretty_print(stream, next_indentation);

    stream << "\n";
    stream << indentation << ")";
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