#include "lab/utils/env.h"

using namespace std::chrono;

namespace lab
{
namespace utils
{
Clock::Clock()
{
    reset();
}
Clock::Clock(double max_frame, double clock_speed /*= 1*/)
    : max_frame(max_frame), clock_speed(clock_speed)
{
    reset();
}

time_point<high_resolution_clock> Clock::now()
{
    return high_resolution_clock::now();
}

void Clock::reset()
{
    time = 0;
    frame = 0;
    epi = 0;
    start_wall_time = now();
    wall_time = 0;
    batch_size = 1;
    opt_step = 0;
}

void Clock::load(utils::DataFrame& train_df)
{
    auto last_clock_vals = train_df.get_last_row({"epi", "time", "wall_time", "opt_step", "frame"});
    epi = last_clock_vals.at<double>(0);
    time = last_clock_vals.at<double>(1);
    wall_time = last_clock_vals.at<double>(2);
    opt_step = last_clock_vals.at<double>(3);
    frame = last_clock_vals.at<double>(4);
    start_wall_time = start_wall_time - duration_cast<nanoseconds>(duration<double>(wall_time));
}

double Clock::get_elapsed_wall_time()
{
    return duration<double>(now() - start_wall_time).count();
}

void Clock::set_batch_size(int64_t size)
{
    batch_size = double(size);
}

void Clock::tick_time()
{
    time += clock_speed;
    frame += clock_speed;
    wall_time = get_elapsed_wall_time();
}

void Clock::tick_epi()
{
    epi += 1;
    time = 0;
}

void Clock::tick_opt_step()
{
    opt_step += batch_size;
}

}
}
