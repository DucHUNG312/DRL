#include "lab/envs/classic_control/cartpole.h"
#include "lab/envs/components.h"

namespace lab
{
namespace envs
{

CartPole::CartPole()
    : Env(utils::SpecLoader::load_default_env_spec("CartPole"))
{
    LAB_CHECK_GE(gravity, 0);
    LAB_CHECK_GE(masscart, 0);
    LAB_CHECK_GE(masspole, 0);
    LAB_CHECK_GE(total_mass, 0);
    LAB_CHECK_GE(length, 0);
    LAB_CHECK_GE(polemass_length, 0);
    LAB_CHECK_GE(force_mag, 0);
    LAB_CHECK_GE(tau, 0);
    LAB_CHECK_GE(theta_threshold_radians, -lab::math::Pi);
    LAB_CHECK_LE(theta_threshold_radians, lab::math::Pi); 
    LAB_CHECK_LT(reset_low, reset_high);

    torch::Tensor high = torch::tensor({x_threshold * 2, lab::math::Max, theta_threshold_radians * 2, lab::math::Max}, torch::kDouble);
    action_spaces_ = spaces::make_discrete_space(2).ptr();
    observation_spaces_ = spaces::make_box_space(-high, high).ptr();
    result_.state = torch::tensor({0, 0, 0, 0}, torch::kDouble);
    env_spec_.is_open = true;

    if (env_spec_.renderer.enabled)
    {
        entity_id = registry_.create("CartPole");
        registry_.add_component<StepResultComponent>(entity_id);
        registry_.add_component<CartPoleActionComponent>(entity_id);
        registry_.add_component<renderer::Velocity>(entity_id);
        registry_.add_component<renderer::ExternalForce>(entity_id, force_mag);
        registry_.add_component<renderer::RigidBody2DComponent>(entity_id, total_mass);
    }
}

void CartPole::reset(uint64_t seed /*= 0*/)
{
    rand_.reset(seed);
    result_.state = rand_.sample_uniform(reset_low * result_.state, reset_high * result_.state);
    steps_beyond_terminated = -1;

    if (env_spec_.renderer.enabled)
    {
        registry_.add_or_replace_component<StepResultComponent>(entity_id, utils::get_data_from_tensor(result_.state), 0, false, false);
        registry_.add_or_replace_component<CartPoleActionComponent>(entity_id, 0);
    }
}

void CartPole::enable_rendering()
{
    env_spec_.renderer.enabled = true;
    render::Renderer::init();
    render();
}

int64_t CartPole::sample()
{
    auto discrete_ptr = action_spaces_->template as<spaces::DiscreteImpl>();
    if (!discrete_ptr) 
    {
        LAB_LOG_FATAL("Discrete space is not initialized correctly.");
        return -1;
    }
    return discrete_ptr->sample();
}

void CartPole::step(int64_t action)
{
    LAB_CHECK(action_spaces_->template as<spaces::DiscreteImpl>()->contains(action));

    double x = result_.state[0].item<double>(); 
    double x_dot = result_.state[1].item<double>();
    double theta = result_.state[2].item<double>();
    double theta_dot = result_.state[3].item<double>();
    double force = (action == 1) ? force_mag : -force_mag;
    double costheta = std::cos(theta);
    double sintheta = std::sin(theta);
    double temp = (force + polemass_length * std::pow(theta_dot, 2) * sintheta) / total_mass;
    double thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0/3.0 - masspole * std::pow(costheta, 2) / total_mass));
    double xacc  = temp - polemass_length * thetaacc * costheta / total_mass;

    if(kinematics_integrator == "euler")
    {
        result_.state[0] = x = x + tau * x_dot;
        result_.state[1] = x_dot = x_dot + tau * xacc;
        result_.state[2] = theta = theta + tau * theta_dot;
        result_.state[3] = theta_dot = theta_dot + tau * thetaacc;
    }
    else
    {
        result_.state[1] = x_dot = x_dot + tau * xacc;
        result_.state[0] = x = x + tau * x_dot;
        result_.state[3] = theta_dot = theta_dot + tau * thetaacc;
        result_.state[2] = theta = theta + tau * theta_dot;
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

    result_.reward = reward;
    result_.terminated = terminated;
    result_.truncated = false;

    // update entity
    if (env_spec_.renderer.enabled)
    {
        registry_.add_or_replace_component<CartPoleActionComponent>(entity_id, action);
        registry_.add_or_replace_component<StepResultComponent>(entity_id, utils::get_data_from_tensor(result_.state), result_.reward, result_.terminated, result_.truncated);
    }
}

void CartPole::render()
{
    if (env_spec_.renderer.enabled)
    {
        render::Renderer::render();
    }
}

void CartPole::close()
{
    env_spec_.is_open = false;
    if (env_spec_.renderer.enabled)
        render::Renderer::shutdown();
}

}
}