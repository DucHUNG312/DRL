#include "lab/envs/classic_control/cartpole.h"

namespace lab
{
namespace envs
{

void CartPole::init()
{
    LAB_CHECK_GE(gravity, 0);
    LAB_CHECK_GE(masscart, 0);
    LAB_CHECK_GE(masspole, 0);
    LAB_CHECK_GE(total_mass, 0);
    LAB_CHECK_GE(length, 0);
    LAB_CHECK_GE(polemass_length, 0);
    LAB_CHECK_GE(force_mag, 0);
    LAB_CHECK_GE(tau, 0);
    LAB_CHECK_GE(theta_threshold_radians, -utils::math::Pi);
    LAB_CHECK_LE(theta_threshold_radians, utils::math::Pi); 
    LAB_CHECK_LT(reset_low, reset_high);

    torch::Tensor high = torch::tensor({x_threshold * 2, utils::math::Max, theta_threshold_radians * 2, utils::math::Max}, torch::kDouble);
    action_space(spaces::Discrete(2));
    observation_space(spaces::Box(-high, high));
    result().state = torch::tensor({0, 0, 0, 0}, torch::kDouble);
    env_options().is_open = true;
}

CartPole::CartPole()
    : ContinuousStateEnv(utils::get_default_env_option("CartPole"))
{
    init();
}

CartPole::CartPole(const utils::EnvOptions& env_options)
    : ContinuousStateEnv(env_options)
{
    init();
}

void CartPole::reset(uint64_t seed /*= 0*/)
{
    result().state = rand().sample_real_uniform(reset_low, reset_high, result().state);
    steps_beyond_terminated = -1;
}

void CartPole::step(const int64_t& action)
{
    LAB_CHECK(action_space().contains(action));

    double x = result().state[0].item<double>(); 
    double x_dot = result().state[1].item<double>();
    double theta = result().state[2].item<double>();
    double theta_dot = result().state[3].item<double>();
    double force = (action == 1) ? force_mag : -force_mag;
    double costheta = std::cos(theta);
    double sintheta = std::sin(theta);
    double temp = (force + polemass_length * std::pow(theta_dot, 2) * sintheta) / total_mass;
    double thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0/3.0 - masspole * std::pow(costheta, 2) / total_mass));
    double xacc  = temp - polemass_length * thetaacc * costheta / total_mass;

    if(kinematics_integrator == "euler")
    {
        result().state[0] = x = x + tau * x_dot;
        result().state[1] = x_dot = x_dot + tau * xacc;
        result().state[2] = theta = theta + tau * theta_dot;
        result().state[3] = theta_dot = theta_dot + tau * thetaacc;
    }
    else
    {
        result().state[1] = x_dot = x_dot + tau * xacc;
        result().state[0] = x = x + tau * x_dot;
        result().state[3] = theta_dot = theta_dot + tau * thetaacc;
        result().state[2] = theta = theta + tau * theta_dot;
    }

    bool terminated = (x < -x_threshold) || (x > x_threshold) || (theta < -theta_threshold_radians) || (theta > theta_threshold_radians);

    double reward;
    if(!terminated)
    {
        reward = 1;
    }
    else if (steps_beyond_terminated != -1)
    {
        steps_beyond_terminated = 0;
        reward = 1;
    }
    else
    {
        if (steps_beyond_terminated == 0)
            LAB_LOG_WARN("You are calling 'step()' even though this environment has already returned done = true. You should always call 'reset()' once you receive 'done = true' -- any further steps are undefined behavior.");
        steps_beyond_terminated = steps_beyond_terminated + 1;
        reward = 0;
    }

    result().reward = reward;
    result().terminated = terminated;
    result().truncated = false;
}

void CartPole::render()
{
    /* Set up */

    /* Render */
    render::Renderer::render();
}

void CartPole::close()
{
    env_options().is_open = false;
    if (env_options_.renderer_enabled)
        render::Renderer::shutdown();
}

c10::intrusive_ptr<ContinuousStateEnv> CartPole::unwrapped()
{
    return c10::make_intrusive<CartPole>(*this);
}

}
}