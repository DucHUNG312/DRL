#pragma once

#include "lab/wrappers/base.h"
#include "lab/spaces/box.h"

namespace lab
{
namespace wrappers
{

template<typename Env>
class ClipAction : public ActionWrapper<Env>
{
public:
    using ActType = typename Env::ActType;

    ClipAction(const c10::intrusive_ptr<Env>& env)
        : ActionWrapper<Env>(std::move(env))
    {
        static_assert(std::is_same_v<ActType, torch::Tensor>);

    }

    ActType action(ActType& act)
    {
        return torch::clip(act, 
            this->env_->get_action_spaces()->template as<spaces::BoxImpl>()->low(), 
            this->env_->get_action_spaces()->template as<spaces::BoxImpl>()->high());
    }
};


}
}