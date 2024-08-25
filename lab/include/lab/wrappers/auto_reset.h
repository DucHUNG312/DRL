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
    using Wrapper<Env>::Wrapper;

    void step(const torch::IValue& action)
    {
        this->unwrapped()->step(action);
        if (this->unwrapped()->result_.terminated || this->unwrapped()->result_.truncated)
            this->unwrapped()->reset();
    }
};


}
}