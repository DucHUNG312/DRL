#pragma once

#include "lab/wrappers/base.h"

namespace lab {
namespace wrappers {

template <typename Env>
class TimeLimit : public Wrapper<Env> {
 public:
  TimeLimit(const spaces::SpaceHolder<Env>& env, int64_t max_frame = -1) : Wrapper<Env>(std::move(env)) {
    if (max_frame == -1 && this->unwrapped()->env_spec_.max_frame != -1)
      max_frame_ = this->unwrapped()->env_spec_.max_frame;
    else if (this->unwrapped()->env_spec_.max_frame == -1)
      this->unwrapped()->env_spec_.max_frame = max_frame;
    max_frame_ = max_frame;
    elapsed_steps_ = 0;
  }

  void reset(uint64_t seed = 0) {
    elapsed_steps_ = 0;
    this->unwrapped()->reset(seed);
  }

  void step(const torch::IValue& action) {
    this->unwrapped()->step(action);
    elapsed_steps_ += 1;

    if (elapsed_steps_ >= max_frame_)
      this->unwrapped()->result_.truncated = true;
  }

 protected:
  int64_t max_frame_;
  int64_t elapsed_steps_;
};

} // namespace wrappers
} // namespace lab