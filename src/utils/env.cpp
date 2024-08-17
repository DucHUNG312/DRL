#include "lab/utils/env.h"
#include "lab/utils/convert.h"
#include "lab/render/render.h"

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
    bool _renderer_enabled,
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
    renderer_enabled = _renderer_enabled;
    screen_width = _screen_width;
    screen_height = _screen_height;
    is_open = _is_open;
}


}
}
