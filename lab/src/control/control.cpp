#include "lab/control/control.h"
#include "lab/utils/control.h"
#include <c10/cuda/CUDACachingAllocator.h>

namespace lab
{
namespace control
{


Session::Session(const utils::LabSpec& lab_spec)
    : spec_(lab_spec)
{
    agent_ = utils::make_agent(lab_spec);
}

void Session::run()
{
    run_rl();
    close();
}

void Session::update_total_reward()
{
    max_total_reward_ = std::max(max_total_reward_, agent_.get_total_reward());
}

void Session::run_rl()
{
    agent_.reset_env();
    auto clock = agent_.get_env_clock();
    while(true)
    {
        if(agent_.is_env_terminated())
            if(clock->frame < clock->max_frame)
            {
                clock->tick_epi();
                agent_.reset_env();
                LAB_LOG_DEBUG("Max Total Reward: {}", max_total_reward_);
            }

        if(clock->frame > clock->max_frame)
            break;

        clock->tick_time();

        // calc action
        torch::Tensor action;
        {
            torch::NoGradGuard no_grad;
            action = agent_.act();
        }

        // do the work
        agent_.step(action);

        // update agent with current state
        agent_.update();

        // update maximum reward so far
        update_total_reward();
    }
}

void Session::close()
{
    agent_.close();
    c10::cuda::CUDACachingAllocator::emptyCache();
    LAB_LOG_DEBUG("Session done");
}

}
}