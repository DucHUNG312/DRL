#include "lab/utils/env.h"
#include "lab/utils/convert.h"

namespace lab
{
namespace utils
{

EnvSpec::EnvSpec(
    const std::string& _id, 
    double _reward_threshold, 
    bool _nondeterministic, 
    int64_t _max_episode_steps, 
    bool _autoreset,
    EnvType _type,
    const std::pair<double, double>& _reward_range,
    bool _render_mode)
{
    LAB_CHECK(!check_env_id_exits(id));
    id = _id;
    reward_threshold = _reward_threshold;
    nondeterministic = _nondeterministic;
    max_episode_steps = _max_episode_steps;
    autoreset = _autoreset;
    type = _type;
    reward_range = _reward_range;
    render_mode = _render_mode;
}

EnvSpec::EnvSpec(const EnvSpec& other)
{
    copy_from(other);
}

EnvSpec::EnvSpec(EnvSpec&& other) noexcept
{
    move_from(std::move(other));
}

bool check_env_id_exits(const std::string& id)
{
    return (env_ids.find(id) != env_ids.end());
}

bool check_env_spec(const EnvSpec& spec)
{
    if(check_env_id_exits(spec.id) || 
        spec.reward_threshold == 0 || 
        spec.max_episode_steps == -1) 
        return false;
    return true;
}

void register_env(const EnvSpec& spec)
{
    LAB_CHECK(check_env_spec(spec));
    env_ids.insert( spec.id );
    env_specs.insert({ spec.id, spec });
}

EnvSpec get_env_spec(const std::string& id)
{
    LAB_CHECK(check_env_id_exits(id));
    return env_specs[id];
}

}
}
