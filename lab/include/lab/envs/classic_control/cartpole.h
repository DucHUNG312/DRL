#pragma once

#include "lab/envs/base.h"

namespace lab {
namespace envs {

class CartPole : public Env {
 public:
  CartPole();
  explicit CartPole(const utils::EnvSpec& env_spec);
  void reset(uint64_t seed = 0) override;
  void step(const torch::Tensor& action) override;
  torch::Tensor sample() override;
  void close() override;
  void render() override;

 public:
  static constexpr const char* name = "CartPole";
  std::string kinematics_integrator = "euler";
  std::string entity_id;
  double gravity = 9.8;
  double masscart = 1.0;
  double masspole = 0.1;
  double total_mass = masspole + masscart;
  double length = 0.5;
  double polemass_length = masspole * length;
  double force_mag = 10.0;
  double tau = 0.02;
  double theta_threshold_radians = 12 * lab::math::Pi / 360;
  double x_threshold = 2.4;
  double reset_low = -0.05;
  double reset_high = 0.05;
  int64_t steps_beyond_terminated = -1;
};

} // namespace envs
} // namespace lab