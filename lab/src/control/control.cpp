#include "lab/control/control.h"
#include "lab/agents/body.h"
#include "lab/utils/net.h"

#include <c10/cuda/CUDACachingAllocator.h>

namespace lab
{
namespace control
{

agents::Agent make_agent(const utils::LabSpec& lab_spec)
{
    std::shared_ptr<envs::Env> env = utils::create_env(lab_spec.env);
    std::shared_ptr<agents::Body> body = std::make_shared<agents::Body>(env, lab_spec.body);
    return agents::Agent(body, lab_spec.agent);
}

Session::Session(const utils::LabSpec& lab_spec)
    : spec_(lab_spec)
{
    agent_ = make_agent(lab_spec);
    eval_env_ = agent_.body()->env();
}

void Session::run()
{
    run_rl();
    close();
}

void Session::run_rl()
{
    eval_env_->reset();
    auto clock = eval_env_->clock();
    torch::Tensor state = eval_env_->result().state;
    double max_total_reward = 0;
    while(true)
    {
        if(eval_env_->result().terminated)
            eval_env_->reset();

        if(clock->frame > clock->max_frame)
            break;

        clock->tick_time();

        torch::Tensor action = agent_.act(state);
        eval_env_->step(action);
        //eval_env_->result().state = state;
        agent_.update(eval_env_->result());
        state = eval_env_->result().state;
        max_total_reward = std::max(max_total_reward, eval_env_->total_reward());
        if(clock->frame % 100 == 0)
        {
            agent_.body()->algorithm()->net()->print_weights();
            LAB_LOG_DEBUG("Max Total Reward: {}", max_total_reward);
        }
            
    }
}

void Session::close()
{
    agent_.close();
    eval_env_->close();
    c10::cuda::CUDACachingAllocator::emptyCache();
    LAB_LOG_DEBUG("Session done");
}

}
}