#include "lab/agents/memory/onpolicy.h"
#include "lab/agents/body.h"

namespace lab
{

namespace agents
{

void OnPolicyReplay::reset()
{
    experiences_.clear();
    for (const auto& key : keys_)
        experiences_.insert(key, c10::impl::GenericList(c10::AnyType::get()));     
}

void OnPolicyReplay::update(const envs::StepResult& result)
{
    add_experience(result);
}

Memory::ExperienceDict OnPolicyReplay::sample()
{
    Memory::ExperienceDict batch = experiences_.copy();
    reset();
    return batch.copy();
}

void OnPolicyReplay::add_experience(const envs::StepResult& result)
{
    most_recent_ = result;

#define ADD_EXPERIENCE(param) experiences_.at(#param).push_back(torch::IValue(result.param))
    ADD_EXPERIENCE(state);
    ADD_EXPERIENCE(next_state);
    ADD_EXPERIENCE(action);
    ADD_EXPERIENCE(reward);
    ADD_EXPERIENCE(terminated);
    ADD_EXPERIENCE(truncated);
#undef ADD_EXPERIENCE

    if(result.terminated) ready(true);
    size(size_ + 1);
    seen_size(seen_size_ + 1);
}


}

}