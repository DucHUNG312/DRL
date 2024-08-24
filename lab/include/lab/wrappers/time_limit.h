#pragma once

#include "lab/wrappers/base.h"

namespace lab
{
namespace wrappers
{

template<typename Env>
class TimeLimit : public Wrapper<Env>
{
public:
    using ActType = typename Env::ActType;

    TimeLimit(const c10::intrusive_ptr<Env>& env, int64_t max_frame = -1)
        : Wrapper<Env>(std::move(env))
    {
        if(max_frame == -1 && this->env_->env_spec_.max_frame != -1)
            max_frame =  this->env_->env_spec_.max_frame;
        else if(this->env_->env_spec_.max_frame == -1)
            this->env_->env_spec_.max_frame = max_frame;
        max_frame_ = max_frame;
        elapsed_steps_ = 0;
    }

    void reset(uint64_t seed = 0)
    {
        elapsed_steps_ = 0;
        this->env_->reset(seed);
    }

    void step(const ActType& action)
    {
        this->env_->step(action);
        elapsed_steps_ += 1;

        if(elapsed_steps_ >= max_frame_)
            this->env_->result_.truncated = true;
    }
protected:
    int64_t max_frame_;
    int64_t elapsed_steps_;
};


}
}