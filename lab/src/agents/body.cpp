#include "lab/agents/body.h"

namespace lab
{

namespace agents
{

utils::BodySpec Body::spec_;

Body::Body(const std::shared_ptr<envs::Env>& env, const utils::BodySpec& spec)
    : env_(env), 
      memory_(std::make_shared<Memory>(shared_from_this(), spec.memory)),
      algorithm_(std::make_shared<Algorithm>(shared_from_this(), spec.algorithm)) 
{
  spec_ = spec; 
}

utils::BodySpec& Body::get_spec()
{
  return spec_;
}

std::shared_ptr<spaces::Space>& Body::get_action_spaces()
{
  return envs::Env::get_action_spaces();
}

}

}