#include "lab/utils/env.h"
#include "lab/utils/convert.h"

namespace lab
{
namespace utils
{
bool check_env_id_exits(const std::string& id)
{
    return (env_ids.find(id) != env_ids.end());
}

bool check_env_option(const EnvOptions& option)
{
    if(check_env_id_exits(option.id) || 
        option.reward_threshold <= 0 || 
        option.max_episode_steps == -1) 
        return false;
    return true;
}

EnvOptions get_default_env_option(const std::string& id)
{
    LAB_CHECK(check_env_id_exits(id));
    return default_env_options[id];
}

EnvOptions::EnvOptions(
    const std::string& _id, 
    double _reward_threshold, 
    bool _nondeterministic,
    uint64_t _seed,
    int64_t _max_episode_steps, 
    bool _auto_reset,
    EnvType _type,
    const std::pair<double, double>& _reward_range,
    const std::string& _render_mode,
    uint64_t _screen_width,
    uint64_t _screen_height,
    bool _is_open)
{
    LAB_CHECK(!check_env_id_exits(id));
    id = _id;
    reward_threshold = _reward_threshold;
    nondeterministic = _nondeterministic;
    seed = _seed;
    max_episode_steps = _max_episode_steps;
    auto_reset = _auto_reset;
    type = _type;
    reward_range = _reward_range;
    render_mode = _render_mode;
    screen_width = _screen_width;
    screen_height = _screen_height;
    is_open = _is_open;
}

EnvOptions::EnvOptions(const EnvOptions& other)
{
    copy_from(other);
}

EnvOptions::EnvOptions(EnvOptions&& other) noexcept
{
    move_from(std::move(other));
}

EnvOptions& EnvOptions::operator=(const EnvOptions& other) 
{
    if (this != &other) 
        copy_from(other);
    return *this;
}
EnvOptions& EnvOptions::operator=(EnvOptions&& other) noexcept 
{
    if (this != &other) 
        move_from(std::move(other));
    return *this;
}

void EnvOptions::copy_from(const EnvOptions& other)
{
    id = other.id;
    reward_threshold = other.reward_threshold;
    nondeterministic = other.nondeterministic;
    seed = other.seed;
    max_episode_steps = other.max_episode_steps;
    auto_reset = other.auto_reset;
    type = other.type;
    reward_range = other.reward_range;
    render_mode = other.render_mode;
    screen_width = other.screen_width;
    screen_height = other.screen_height;
    is_open = other.is_open;
}

void EnvOptions::move_from(EnvOptions&& other) noexcept
{
    id = std::move(other.id);
    reward_threshold = std::move(other.reward_threshold);
    nondeterministic = std::move(other.nondeterministic);
    seed = std::move(other.seed);
    max_episode_steps = std::move(other.max_episode_steps);
    auto_reset = std::move(other.auto_reset);
    type = std::move(other.type);
    reward_range = std::move(other.reward_range);
    render_mode = std::move(other.render_mode);
    screen_width = std::move(other.screen_width);
    screen_height = std::move(other.screen_height);
    is_open = std::move(other.is_open);
}

}
}
