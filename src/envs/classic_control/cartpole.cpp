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
    state(torch::tensor({0, 0, 0, 0}, torch::kDouble));
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

torch::Tensor CartPole::reset(uint64_t seed /*= 0*/)
{
    state(rand().sample_real_uniform(reset_low, reset_high, state()));
    steps_beyond_terminated = -1;
    if(env_options().render_mode == "human") 
        render();
    return state();
}

ContinuousResult CartPole::step(const int64_t& action)
{
    LAB_CHECK(action_space().contains(action));

    double x = state()[0].item<double>(); 
    double x_dot = state()[1].item<double>();
    double theta = state()[2].item<double>();
    double theta_dot = state()[3].item<double>();
    double force = (action == 1) ? force_mag : -force_mag;
    double costheta = std::cos(theta);
    double sintheta = std::sin(theta);
    double temp = (force + polemass_length * std::pow(theta_dot, 2) * sintheta) / total_mass;
    double thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0/3.0 - masspole * std::pow(costheta, 2) / total_mass));
    double xacc  = temp - polemass_length * thetaacc * costheta / total_mass;

    if(kinematics_integrator == "euler")
    {
        state()[0] = x = x + tau * x_dot;
        state()[1] = x_dot = x_dot + tau * xacc;
        state()[2] = theta = theta + tau * theta_dot;
        state()[3] = theta_dot = theta_dot + tau * thetaacc;
    }
    else
    {
        state()[1] = x_dot = x_dot + tau * xacc;
        state()[0] = x = x + tau * x_dot;
        state()[3] = theta_dot = theta_dot + tau * thetaacc;
        state()[2] = theta = theta + tau * theta_dot;
    }

    bool done = (x < -x_threshold) || (x > x_threshold) || (theta < -theta_threshold_radians) || (theta > theta_threshold_radians);

    double reward;
    if(!done)
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

    if(env_options().render_mode == "human") 
        render();

    return ContinuousResult(state(), reward, done, false);
}

void CartPole::render()
{
    LAB_UNIMPLEMENTED;
}

void CartPole::close()
{
    if(env_options().render_mode == "human") 
        LAB_UNIMPLEMENTED;
    env_options().is_open = false;
}

c10::intrusive_ptr<ContinuousStateEnv> CartPole::unwrapped()
{
    return c10::make_intrusive<CartPole>(*this);
}

}
}