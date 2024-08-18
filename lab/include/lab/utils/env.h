#pragma once

#include "lab/core.h"
#include "lab/utils/dataframe.h"

namespace lab
{

namespace utils
{

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
    os << "state: " << result.state << "\n";
    os << "reward: " << result.reward << "\n";
    os << "terminated: " << (result.terminated ? "true" : "false") << "\n";
    os << "truncated: " << (result.truncated ? "true" : "false");
    return os;
}

class Clock
{
public:
    Clock();
    Clock(double max_frame, double clock_speed = 1);
    Clock(const Clock& other) = default;
    Clock(Clock&& other) noexcept = default;
    Clock& operator=(const Clock& other) = default;
    Clock& operator=(Clock&& other) noexcept = default;
    virtual ~Clock() = default;

    static std::chrono::time_point<std::chrono::high_resolution_clock> now();

    void reset();
    void load(utils::DataFrame& train_df);
    double get_elapsed_wall_time();
    void set_batch_size(int64_t size);
    void tick_time();
    void tick_epi();
    void tick_opt_step();
public:
    double epi;
    double time;
    double wall_time;
    double batch_size;
    double opt_step;
    double frame;
    double max_frame;
    double clock_speed;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_wall_time;
};

}
}