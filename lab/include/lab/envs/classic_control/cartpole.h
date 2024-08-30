#pragma once

#include "lab/core.h"
#include "lab/envs/base.h"

namespace lab
{
namespace envs
{

class CartPole : public Env
{
public:
    CartPole();
    void reset(uint64_t seed = 0);
    void step(const torch::Tensor& action);
    torch::Tensor sample();
    void close();
    void render();
    void enable_rendering();
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
    double theta_threshold_radians = 12 * 2 * lab::math::Pi / 360;
    double x_threshold = 2.4;
    int64_t steps_beyond_terminated = -1;
    double reset_low = -0.05;
    double reset_high = 0.05;
    std::string entity_id;
};

}
}