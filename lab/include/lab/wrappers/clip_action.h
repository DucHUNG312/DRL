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
    using ActionWrapper<Env>::ActionWrapper;

    torch::IValue action(torch::IValue& act)
    {
        LAB_CHECK(act.isTensor());
        return torch::clip(act, 
            this->unwrapped()->get_action_spaces()->template as<spaces::Box>()->low(), 
            this->unwrapped()->get_action_spaces()->template as<spaces::Box>()->high());
    }
};


}
}