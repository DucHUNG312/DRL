#pragma once

#include "lab/core.h"
#include "lab/utils/spec.h"
#include "lab/utils/typetraits.h"

namespace lab
{

namespace envs
{

class Env;
class CartPole;

}

namespace utils
{

using Envs = types_t<envs::CartPole>;
constexpr named_factory_t<std::shared_ptr<envs::Env>, shared_ptr_maker, Envs> EnvFactory;

std::shared_ptr<envs::Env> create_env(const EnvSpec& spec);

class Clock
{
public:
    Clock();
    
    Clock(int64_t max_frame, int64_t clock_speed = 1);
    
    Clock(const Clock& other) = default;
    
    Clock(Clock&& other) noexcept = default;
    
    Clock& operator=(const Clock& other) = default;
    
    Clock& operator=(Clock&& other) noexcept = default;
    
    virtual ~Clock() = default;

    void reset();
    
    void load(double time, double elapsed_wall_time, int64_t epi, int64_t opt_step, int64_t frame);
    
    double get_elapsed_wall_time();
    
    void set_batch_size(int64_t size);
    
    void tick_time();
    
    void tick_epi();
    
    void tick_opt_step();

    static std::chrono::time_point<std::chrono::high_resolution_clock> now();
public:
    double time;
    double elapsed_wall_time;
    int64_t epi;
    int64_t batch_size;
    int64_t opt_step;
    int64_t frame;
    int64_t max_frame;
    int64_t clock_speed;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_wall_time;
};

}
}