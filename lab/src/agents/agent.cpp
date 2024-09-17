#include "lab/agents/agent.h"

namespace lab {

namespace agents {

Agent::Agent(const std::shared_ptr<Body>& body, const utils::AgentSpec& spec) : body_(body), spec_(std::move(spec)) {}

torch::Tensor Agent::act() {
  torch::NoGradGuard no_grad;
  LAB_CHECK(body_ != nullptr, "Body pointer is nullptr");
  return body_->act();
}

void Agent::update() {
  body_->update();
}

void Agent::reset_env() {
  body_->reset_env();
}

double Agent::get_total_reward() const {
  return body_->get_total_reward();
}

bool Agent::is_env_terminated() const {
  return body_->is_env_terminated();
}

void Agent::step(const torch::Tensor& act) {
  body_->step(act);
}

torch::Tensor Agent::get_result_state() const {
  return body_->get_result_state();
}

std::shared_ptr<utils::Clock> Agent::get_env_clock() const {
  return body_->get_env_clock();
}

void Agent::close() {
  body_->close_env();
  // TODO: save();
}

void Agent::save(torch::serialize::OutputArchive& archive) const {
  body_->save(archive);
}

void Agent::load(torch::serialize::InputArchive& archive) {
  body_->load(archive);
}

torch::serialize::OutputArchive& operator<<(
    torch::serialize::OutputArchive& archive,
    const std::shared_ptr<Agent>& agent) {
  LAB_CHECK(agent != nullptr, "Agent pointer is nullptr");
  agent->save(archive);
  return archive;
}

torch::serialize::InputArchive& operator>>(
    torch::serialize::InputArchive& archive,
    const std::shared_ptr<Agent>& agent) {
  LAB_CHECK(agent != nullptr, "Agent pointer is nullptr");
  agent->load(archive);
  return archive;
}

} // namespace agents

} // namespace lab