#include "lab/agents/memory/onpolicy.h"

namespace lab {

namespace agents {

void OnPolicyReplay::update(const utils::StepResult& result) {
  add_experience(result);
}

Memory::ExperienceDict OnPolicyReplay::sample() {
  Memory::ExperienceDict batch = experiences_.copy();
  reset();
  return batch.copy();
}

void OnPolicyReplay::add_experience(const utils::StepResult& result) {
  most_recent_ = result;

#define ADD_EXPERIENCE(param) experiences_.at(#param).push_back(torch::IValue(result.param))
  ADD_EXPERIENCE(state);
  // ADD_EXPERIENCE(next_state);
  ADD_EXPERIENCE(action);
  ADD_EXPERIENCE(reward);
  ADD_EXPERIENCE(terminated);
  ADD_EXPERIENCE(truncated);
#undef ADD_EXPERIENCE

  if (result.terminated)
    ready_ = true;

  size_ = size_ + 1;
  seen_size_ = seen_size_ + 1;
}

} // namespace agents

} // namespace lab