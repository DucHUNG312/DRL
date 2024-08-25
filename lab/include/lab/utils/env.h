#pragma once

#include "lab/core.h"

namespace lab
{

namespace utils
{

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
    void load();
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