#include "lab/agents/base.h"

namespace lab
{

namespace agents
{

Agent::Agent(const std::shared_ptr<Body>& body, const utils::AgentSpec& spec)
    : body_(body), spec_(std::move(spec))
{}

torch::Tensor Agent::act(const torch::Tensor& state)
{
    torch::NoGradGuard no_grad;
    LAB_CHECK(body_ != nullptr);
    return body_->act(state);
}

torch::Tensor Agent::update(const envs::StepResult& result)
{
    body_->update(result);
    torch::Tensor loss = body_->train();
    if(loss.defined())
        body_->algorithm()->loss(loss);
    body_->algorithm()->update();
    return loss;
}

void Agent::close()
{
    // TODO: save();
}

void Agent::save(torch::serialize::OutputArchive& archive) const
{
    body_->save(archive);
}

void Agent::load(torch::serialize::InputArchive& archive)
{
    body_->load(archive);
}

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Agent>& agent)
{
    LAB_CHECK(agent != nullptr);
    agent->save(archive);
    return archive;
}

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Agent>& agent)
{
    LAB_CHECK(agent != nullptr);
    agent->load(archive);
    return archive;
}

}

}