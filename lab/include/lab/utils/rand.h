#pragma once

#include "lab/common/common.h"

namespace lab
{
namespace utils
{
class Rand
{
public:
    Rand(uint64_t seed);
    LAB_DEFAULT_CONSTRUCT(Rand);

    void reset(uint64_t seed = 0);

    void set_current_seed(uint64_t seed);

    uint64_t current_seed();

    void set_state(const torch::Tensor& new_state);

    torch::Tensor get_state() const;

    int64_t choice(const torch::Tensor& prob, std::optional<double> u = std::nullopt);

    torch::Tensor sample_uniform(const torch::Tensor& a, const torch::Tensor& b);

    torch::Tensor sample_gaussian(double mu, double sigma, torch::IntArrayRef shape);

    torch::Tensor gen_int_permutation(int64_t n);

    int64_t sample_int_uniform(int64_t a, int64_t b);

    double sample_double_uniform(double a, double b);

    static double rand();
private:
    torch::Generator generator_;
};    

}
}