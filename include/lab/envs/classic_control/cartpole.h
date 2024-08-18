#pragma once

#include "lab/core.h"
#include "lab/envs/base.h"

namespace lab
{
namespace envs
{

class CartPole : public ContinuousStateEnv
{
public:
    CartPole();
    CartPole(const utils::EnvSpec& env_spec);
    virtual ~CartPole() = default;
    virtual void reset(uint64_t seed = 0) override;
    virtual void step(const int64_t& action) override;
    virtual void close() override;
    virtual void render() override;
    virtual c10::intrusive_ptr<ContinuousStateEnv> unwrapped() override;
public:
    double gravity = 9.8;
    double masscart = 1.0;
    double masspole = 0.1;
    double total_mass = masspole + masscart;
    double length = 0.5;
    double polemass_length = masspole * length;
    double force_mag = 10.0;
    double tau = 0.02;
    std::string kinematics_integrator = "euler";
    double theta_threshold_radians = 12 * 2 * utils::math::Pi / 360;
    double x_threshold = 2.4;
    int64_t steps_beyond_terminated = -1;
    double reset_low = -0.05;
    double reset_high = 0.05;
private: 
    void init();
};

}
}