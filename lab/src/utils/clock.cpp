#include "lab/utils/clock.h"

namespace lab {
namespace utils {

using namespace std::chrono;

Clock::Clock() {
  reset();
}
Clock::Clock(int64_t max_frame, int64_t clock_speed /*= 1*/) : max_frame(max_frame), clock_speed(clock_speed) {
  reset();
}

time_point<high_resolution_clock> Clock::now() {
  return high_resolution_clock::now();
}

void Clock::reset() {
  time = 0;
  frame = 0;
  epi = 0;
  start_wall_time = now();
  elapsed_wall_time = 0;
  batch_size = 1;
  opt_step = 0;
}

void Clock::load(double time_, double elapsed_wall_time_, int64_t epi_, int64_t opt_step_, int64_t frame_) {
  time = time_;
  elapsed_wall_time = elapsed_wall_time_;
  epi = epi_;
  opt_step = opt_step_;
  frame = frame_;
  start_wall_time = start_wall_time - duration_cast<nanoseconds>(duration<double>(elapsed_wall_time_));
}

double Clock::get_elapsed_wall_time() {
  return duration<double>(now() - start_wall_time).count();
}

void Clock::set_batch_size(int64_t size) {
  batch_size = size;
}

void Clock::tick_time() {
  time += clock_speed;
  frame += clock_speed;
  elapsed_wall_time = get_elapsed_wall_time();
}

void Clock::tick_epi() {
  epi += 1;
  time = 0;
}

void Clock::tick_opt_step() {
  opt_step += batch_size;
}

} // namespace utils

} // namespace lab