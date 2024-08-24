#pragma once

#include "lab/wrappers/base.h"

namespace lab
{
namespace wrappers
{

template<typename Env>
class AutoResetWrapper : public Wrapper<Env>
{
public:
    using ActType = typename Env::ActType;

    AutoResetWrapper(const c10::intrusive_ptr<Env>& env)
        : Wrapper<Env>(std::move(env))
    {}

    void step(const ActType& action)
    {
        this->env_->step(action);
        if (this->env_->result_.terminated || this->env_->result_.truncated)
            this->env_->reset();
    }
};


}
}