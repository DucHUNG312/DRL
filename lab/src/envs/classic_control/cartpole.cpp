#include "lab/envs/classic_control/cartpole.h"
#include "lab/envs/components.h"
#include "lab/spaces/discrete.h"
#include "lab/spaces/box.h"
#include "lab/utils/convert.h"

namespace lab
{
namespace envs
{

CartPole::CartPole()
    : CartPole(utils::SpecLoader::load_default_env_spec(name)) {}

CartPole::CartPole(const utils::EnvSpec& env_spec)
    : Env(env_spec)
{
    LAB_CHECK_GE(gravity, 0);
    LAB_CHECK_GE(masscart, 0);
    LAB_CHECK_GE(masspole, 0);
    LAB_CHECK_GE(total_mass, 0);
    LAB_CHECK_GE(length, 0);
    LAB_CHECK_GE(polemass_length, 0);
    LAB_CHECK_GE(force_mag, 0);
    LAB_CHECK_GE(tau, 0);
    LAB_CHECK_GE(theta_threshold_radians, -math::Pi);
    LAB_CHECK_LE(theta_threshold_radians, math::Pi); 
    LAB_CHECK_LT(reset_low, reset_high);

    torch::Tensor high = torch::tensor({x_threshold * 2, math::Max, theta_threshold_radians * 2, math::Max}, torch::kDouble);
    action_spaces_ = spaces::make_discrete_space_any(2);
    observation_spaces_ = spaces::make_box_space_any(-high, high);
    result_.state = torch::tensor({0, 0, 0, 0}, torch::kDouble);
    result_.next_state = result_.state;
    is_open_ = true;

    // if (env_spec.renderer.enabled)
    // {
    //     entity_id = registry_.create(name);
    //     registry_.add_component<StepResultComponent>(entity_id);
    //     registry_.add_component<CartPoleActionComponent>(entity_id);
    //     registry_.add_component<renderer::Velocity>(entity_id);
    //     registry_.add_component<renderer::ExternalForce>(entity_id, force_mag);
    //     registry_.add_component<renderer::RigidBody2DComponent>(entity_id, total_mass);
    // }
}

void CartPole::reset(uint64_t seed /*= 0*/)
{
    rand_.reset(seed);
    total_reward_ = 0;
    result_.state = rand_.sample_uniform(reset_low * result_.state, reset_high * result_.state).to(torch::kDouble);
    result_.next_state = result_.state;
    result_.reward = 0;
    result_.terminated = false;
    result_.truncated = false;
    steps_beyond_terminated = -1;

    // if (env_spec_.renderer.enabled)
    // {
    //     registry_.add_or_replace_component<StepResultComponent>(entity_id, utils::get_data_from_tensor(result_.state), 0, false, false);
    //     registry_.add_or_replace_component<CartPoleActionComponent>(entity_id, 0);
    // }
}

torch::Tensor CartPole::sample()
{
    auto discrete_ptr = action_spaces_->template ptr<spaces::Discrete>();
    if (!discrete_ptr) 
    {
        LAB_LOG_FATAL("Discrete space is not initialized correctly.");
        return torch::Tensor();
    }
    return discrete_ptr->sample();
}

void CartPole::step(const torch::Tensor& action)
{
    action.to(torch::kInt64);
    
    LAB_CHECK(action_spaces_->template ptr<spaces::Discrete>()->contains(action));

    result_.action = action;
    result_.state = result_.next_state;


    double x = result_.state[0].item<double>(); 
    double x_dot = result_.state[1].item<double>();
    double theta = result_.state[2].item<double>();
    double theta_dot = result_.state[3].item<double>();
    double force = (action.item<int64_t>() == 1) ? force_mag : -force_mag;
    double costheta = std::cos(theta);
    double sintheta = std::sin(theta);
    double temp = (force + polemass_length * std::pow(theta_dot, 2) * sintheta) / total_mass;
    double thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0/3.0 - masspole * std::pow(costheta, 2) / total_mass));
    double xacc  = temp - polemass_length * thetaacc * costheta / total_mass;

    if(kinematics_integrator == "euler")
    {
        result_.next_state[0] = x = x + tau * x_dot;
        result_.next_state[1] = x_dot = x_dot + tau * xacc;
        result_.next_state[2] = theta = theta + tau * theta_dot;
        result_.next_state[3] = theta_dot = theta_dot + tau * thetaacc;
    }
    else
    {
        result_.next_state[1] = x_dot = x_dot + tau * xacc;
        result_.next_state[0] = x = x + tau * x_dot;
        result_.next_state[3] = theta_dot = theta_dot + tau * thetaacc;
        result_.next_state[2] = theta = theta + tau * theta_dot;
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
            LAB_LOG_WARN("You are calling 'step()' even though this environment has already returned terminated = true. You should always call 'reset()' once you receive 'terminated = true' -- any further steps are undefined behavior.");
        steps_beyond_terminated += 1;
        reward = 0;
    }

    result_.reward = reward;
    result_.terminated = terminated;
    result_.truncated = false;

    total_reward_ += reward;

    // update entity
    // if (env_spec_.renderer.enabled)
    // {
    //     registry_.add_or_replace_component<CartPoleActionComponent>(entity_id, action.item<int64_t>());
    //     registry_.add_or_replace_component<StepResultComponent>(entity_id, utils::get_data_from_tensor(result_.state), result_.reward, result_.terminated, result_.truncated);
    // }
}

void CartPole::render()
{
    // if (env_spec_.renderer.enabled)
    // {
    //     render::Renderer::render();
    // }
}

void CartPole::close()
{
    is_open_ = false;
    // if (env_spec_.renderer.enabled)
    //     render::Renderer::shutdown();
}

}
}