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
class StepResult
{
    T next_state;
    double reward = 0;
    bool terminated = false; 
    bool truncated = false;
public:
    StepResult() = default;
    StepResult(const T& next_state, double reward, bool terminated, bool truncated)
        : next_state(next_state), reward(reward), terminated(terminated), truncated(truncated) {}
    StepResult(const StepResult& other)
    {
        next_state = other.next_state; 
        reward = other.reward; 
        terminated = other.terminated; 
        truncated = other.truncated; 
    }
    StepResult(StepResult&& other) noexcept
    {
        next_state = std::move(other.next_state); 
        reward = std::move(other.reward); 
        terminated = std::move(other.terminated); 
        truncated = std::move(other.truncated); 

    }
    StepResult& operator=(const StepResult& other) 
    {
        if (this != &other) 
        {
            next_state = other.next_state;
            reward = other.reward;
            terminated = other.terminated;
            truncated = other.truncated;
        }
        return *this;
    }
    StepResult& operator=(StepResult&& other) noexcept 
    {
        if (this != &other) 
        {
            next_state = std::move(other.next_state);
            reward = std::move(other.reward);
            terminated = std::move(other.terminated);
            truncated = std::move(other.truncated);
        }
        return *this;
    }
    virtual ~StepResult() = default;


    bool operator==(const StepResult& other) const 
    {
        return next_state == other.next_state &&
               reward == other.reward &&
               terminated == other.terminated &&
               truncated == other.truncated;
    }

    bool operator==(StepResult& other) 
    {
        return next_state == other.next_state &&
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
    std::string render_mode = "None";
    uint64_t screen_width = 0;
    uint64_t screen_height = 0;
    bool is_open = false;
public:
    EnvOptions() = default;
    EnvOptions(
        const std::string& id, 
        double reward_threshold, 
        bool nondeterministic, 
        uint64_t seed,
        int64_t max_episode_steps, 
        bool auto_reset,
        EnvType type,
        const std::pair<double, double>& reward_range,
        const std::string& render_mode,
        uint64_t screen_width,
        uint64_t screen_height,
        bool is_open);
    EnvOptions(const EnvOptions& other);
    EnvOptions(EnvOptions&& other) noexcept;
    EnvOptions& operator=(const EnvOptions& other);
    EnvOptions& operator=(EnvOptions&& other) noexcept;
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
    "CartPole",   EnvOptions("CartPole", 180, true, 0, 100, false, EnvType::CONTINUOUS_STATE, {0,200}, "None", 600, 400, false)
}
};

bool check_env_id_exits(const std::string& id);

bool check_env_option(const EnvOptions& option);

EnvOptions get_default_env_option(const std::string& id);

}
}