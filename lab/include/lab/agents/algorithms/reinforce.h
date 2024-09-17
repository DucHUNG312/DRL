#pragma once

#include "lab/agents/algorithms/base.h"

namespace lab {

namespace agents {

class Reinforce : public Algorithm {
 public:
  static constexpr const char* name = "Reinforce";

 public:
  using Algorithm::Algorithm;
  using Algorithm::ExperienceDict;

  torch::Tensor train(const ExperienceDict& experiences) override;

  void update(const torch::Tensor& loss) override;

  torch::Tensor act(const torch::Tensor& state) override;

  torch::Tensor calc_ret_advs(const ExperienceDict& experiences);

  torch::Tensor calc_policy_loss(
      const ExperienceDict& experiences,
      const torch::Tensor& states,
      const torch::Tensor& advs);
};

} // namespace agents

} // namespace lab