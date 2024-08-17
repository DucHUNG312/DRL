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
    NONE,
};

template<typename T>
struct StepResult
{
    T state;
    double reward = 0;
    bool terminated = false; 
    bool truncated = false;

    StepResult() = default;
    StepResult(const T& state, double reward, bool terminated, bool truncated)
        : state(state), reward(reward), terminated(terminated), truncated(truncated) {}
    StepResult(const StepResult& other) = default;
    StepResult(StepResult&& other) noexcept = default;
    StepResult& operator=(const StepResult& other) = default;
    StepResult& operator=(StepResult&& other) noexcept = default;
    virtual ~StepResult() = default;

    bool operator==(const StepResult& other) const 
    {
        return state == other.state &&
               reward == other.reward &&
               terminated == other.terminated &&
               truncated == other.truncated;
    }

    bool operator==(StepResult& other) 
    {
        return state == other.state &&
               reward == other.reward &&
               terminated == other.terminated &&
               truncated == other.truncated;
    }

    bool operator!=(const StepResult& other) const 
    {
        return !(*this == other);
    }

    bool operator!=(StepResult& other) 
    {
        return !(*this == other);
    }
};

template<typename T>
LAB_FORCE_INLINE std::ostream& operator<<(std::ostream& os, const StepResult<T>& result)
{
    os << "result:\n";
    os << "\tstate: " << result.state << "\n";
    os << "\treward: " << result.reward << "\n";
    os << "\tterminated: " << (result.terminated ? "true" : "false") << "\n";
    os << "\ttruncated: " << (result.truncated ? "true" : "false");
    return os;
}

struct EnvOptions
{
    std::string id;
    double reward_threshold = 0;
    bool nondeterministic = true;
    uint64_t seed = 0;
    int64_t max_episode_steps = -1;
    bool auto_reset = false;
    EnvType type = EnvType::NONE;
    std::pair<double, double> reward_range = { std::numeric_limits<double>::min(), std::numeric_limits<double>::max() };
    bool renderer_enabled = false;
    uint64_t screen_width = 0;
    uint64_t screen_height = 0;
    bool is_open = false;
public:
    EnvOptions() = default;
    EnvOptions(
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
        bool _is_open);
    EnvOptions(const EnvOptions& other) = default;
    EnvOptions(EnvOptions&& other) noexcept = default;
    EnvOptions& operator=(const EnvOptions& other) = default;
    EnvOptions& operator=(EnvOptions&& other) noexcept = default;
    virtual ~EnvOptions() = default;
private:
    void copy_from(const EnvOptions& other);
    void move_from(EnvOptions&& other) noexcept;
};

static std::unordered_set<std::string> env_ids = 
{
   "CartPole",
};

static std::unordered_map<std::string, EnvOptions> default_env_options = 
{
{   
    "CartPole",   EnvOptions("CartPole", 180, true, 0, 100, false, EnvType::CONTINUOUS_STATE, {0,200}, false, 1280, 720, false)
}
};

bool check_env_id_exits(const std::string& id);

bool check_env_option(const EnvOptions& option);

EnvOptions get_default_env_option(const std::string& id);

}
}