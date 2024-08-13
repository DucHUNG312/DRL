#pragma once

#include "lab/core.h"

namespace lab
{

namespace utils
{

enum class EnvType
{
    FINITE,
    CONTINUOUS,
    CONTINUOUS_STATE,
    CONTINUOUS_ACT,
};

struct EnvSpec
{
    std::string id;
    double reward_threshold = 0;
    bool nondeterministic = true;
    int64_t max_episode_steps = -1;
    bool autoreset = false;
    EnvType type;
    std::pair<double, double> reward_range = { std::numeric_limits<double>::min(), std::numeric_limits<double>::max() };
    bool render_mode = false;

    EnvSpec() = default;
    EnvSpec(
        const std::string& id, 
        double reward_threshold, 
        bool nondeterministic, 
        int64_t max_episode_steps, 
        bool autoreset,
        EnvType type,
        const std::pair<double, double>& reward_range,
        bool render_mode);

    EnvSpec(const EnvSpec& other);

    EnvSpec(EnvSpec&& other) noexcept;

    EnvSpec& operator=(const EnvSpec& other) 
    {
        if (this != &other) 
            copy_from(other);
        return *this;
    }

    EnvSpec& operator=(EnvSpec&& other) noexcept 
    {
        if (this != &other) 
            move_from(std::move(other));
        return *this;
    }
private:
    void copy_from(const EnvSpec& other)
    {
        id = other.id;
        reward_threshold = other.reward_threshold;
        nondeterministic = other.nondeterministic;
        max_episode_steps = other.max_episode_steps;
        autoreset = other.autoreset;
        type = other.type;
        reward_range = other.reward_range;
        render_mode = other.render_mode;
    }

    void move_from(EnvSpec&& other) noexcept
    {
        id = std::move(other.id);
        reward_threshold = std::move(other.reward_threshold);
        nondeterministic = std::move(other.nondeterministic);
        max_episode_steps = std::move(other.max_episode_steps);
        autoreset = std::move(other.autoreset);
        type = std::move(other.type);
        reward_range = std::move(other.reward_range);
        render_mode = std::move(other.render_mode);
    }
};

static std::unordered_set<std::string> env_ids = 
{
   "CartPole-v0",
};

static std::unordered_map<std::string, EnvSpec> env_specs = 
{
/*       ID                       id       |  reward threshold  |  nondeterministic  |  max episode  |  autoreset  |             type              |  reward_range  |  render   */
{   "CartPole-v0",   EnvSpec("CartPole-v0",         180,                 true,              100,          false,       EnvType::CONTINUOUS_STATE,       {0,200},         false     )}
};

bool check_env_id_exits(const std::string& id);

bool check_env_spec(const EnvSpec& spec);

void register_env(const EnvSpec& spec);

EnvSpec get_env_spec(const std::string& id);

}
}