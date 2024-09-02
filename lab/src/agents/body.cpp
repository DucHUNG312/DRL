#include "lab/agents/body.h"
#include "lab/utils/algorithm.h"
#include "lab/utils/memory.h"
#include "lab/utils/net.h"
#include "lab/utils/policy.h"

namespace lab
{

namespace agents
{

Body::Body(const std::shared_ptr<envs::Env>& env, const utils::BodySpec& spec)
    : env_(env), spec_(std::move(spec))
{
  init_algorithm_specs(spec.algorithm);
  init_memory_spec(spec.memory);
}

void Body::init_algorithm_specs(const utils::AlgorithmSpec& spec)
{
    algorithm_ = utils::create_algorithm(std::move(spec));
    algorithm_->env(env_);
    algorithm_->net(utils::create_net(spec_.net, env_->get_state_dim(), env_->get_action_dim()));
    algorithm_->optimizer(utils::create_optim(algorithm_->net()->spec().optim_spec.name, algorithm_->net()->parameters()));
    algorithm_->lrscheduler(utils::create_lr_schedular(algorithm_->optimizer(), algorithm_->net()->spec().lr_scheduler_spec));
    algorithm_->policy(utils::create_action_policy(spec.action_policy));
    algorithm_->explore_var_scheduler(utils::VarScheduler(spec.explore_spec));
    algorithm_->entropy_coef_scheduler(utils::VarScheduler(spec.entropy_spec));
}

void Body::init_memory_spec(const utils::MemorySpec& spec)
{
    memory_ = utils::create_memory(std::move(spec));
}

void Body::update()
{
    memory_->update(env_->result().clone().to(torch::kCPU)); // store memory in CPU
    if(memory_->ready() /*&& memory_->size() == algorithm_->spec().training_frequency*/)
      algorithm_->to_train(true);
    
    torch::Tensor loss = train();
    algorithm_->update(loss);
}

torch::Tensor Body::act(const torch::Tensor& state)
{
    torch::Tensor action = algorithm_->act(state);
    return action;
}

torch::Tensor Body::train() 
{
    if(algorithm_->to_train())
    {
        Body::ExperienceDict experiences = sample();
        env_->clock()->set_batch_size(memory_->size());
        return algorithm_->train(experiences);
    }
    return torch::Tensor();
}

Body::ExperienceDict Body::sample()
{
    return memory_->sample();
}

void Body::save(torch::serialize::OutputArchive& archive) const
{
  algorithm_->save(archive);
  memory_->save(archive);
}

void Body::load(torch::serialize::InputArchive& archive)
{
  memory_->load(archive);
  algorithm_->load(archive);
}

torch::Tensor Body::get_loss() const
{
  return algorithm_->loss();
}

void Body::close_env()
{
  env_->close();
}

void Body::reset_env()
{
  env_->reset();
}

double Body::get_total_reward() const
{
  return env_->total_reward();
}

bool Body::is_env_terminated() const
{
  return env_->result().terminated;
}

void Body::step(const torch::Tensor& act)
{
  env_->step(act);
}

torch::Tensor Body::get_result_state() const
{
  return env_->result().state.clone();
}

std::shared_ptr<utils::Clock> Body::get_env_clock() const
{
  return env_->clock();
}

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Body>& body)
{
  LAB_CHECK(body != nullptr);
  body->save(archive);
  return archive;
}

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Body>& body)
{
  LAB_CHECK(body != nullptr);
  body->load(archive);
  return archive;
}

}

}