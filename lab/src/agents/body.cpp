#include "lab/agents/body.h"

namespace lab
{

namespace agents
{

Body::Body(const std::shared_ptr<envs::Env>& env, const utils::BodySpec& spec)
    : env_(env), 
      spec_(spec),
      memory_(std::make_shared<Memory>(shared_from_this(), spec.memory)),
      algorithm_(std::make_shared<Algorithm>(shared_from_this(), spec.algorithm)) {}

std::shared_ptr<spaces::AnySpace>& Body::get_action_spaces()
{
  return env_->get_action_spaces();
}

}

}