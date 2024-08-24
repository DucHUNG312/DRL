#include "lab/agents/agents.h"

namespace lab
{

namespace agents
{

Body::Body(const std::shared_ptr<envs::Env>& env)
    : env_(env)
{
    // train_df_.load_df_columns({"epi", "time", "wall_time", "opt_step", "frame", "fps", "total_reward", "total_reward_ma", "loss", "lr", "explore_var", "entropy_coef", "entropy", "grad_norm"});
}

Agent::Agent(const std::shared_ptr<Body>& body)
{
    body_ = body;
    body_->agent(shared_from_this());
    // spec(body_->spec().agent);
    // body_->memory(spec_.memory);
}

}

}