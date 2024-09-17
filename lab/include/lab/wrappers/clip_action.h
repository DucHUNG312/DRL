#pragma once

#include "lab/spaces/box.h"
#include "lab/wrappers/base.h"

namespace lab {
namespace wrappers {

template <typename Env>
class ClipAction : public ActionWrapper<Env> {
 public:
  using ActionWrapper<Env>::ActionWrapper;

  torch::IValue action(torch::IValue& act) {
    LAB_CHECK(act.isTensor(), "Action must has type torch::Tensor");
    return torch::clip(
        act,
        this->unwrapped()->get_action_spaces()->template ptr<spaces::Box>()->low(),
        this->unwrapped()->get_action_spaces()->template ptr<spaces::Box>()->high());
  }
};

} // namespace wrappers
} // namespace lab