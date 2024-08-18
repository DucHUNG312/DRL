#pragma once

#include "lab/core.h"
#include "lab/utils/no_grad.h"

namespace lab
{
namespace utils
{
class Rand
{
public:
    torch::Generator generator;
    uint64_t seed;
public:
    Rand(uint64_t seed = 0);
    ~Rand() = default;
    Rand(const Rand& other) = default;
    Rand(Rand&& other) noexcept = default;
    Rand& operator=(const Rand& other) = default;
    Rand& operator=(Rand&& other) noexcept = default;

    void set_seed(uint64_t seed);

    int64_t choice(const torch::Tensor& prob, double u = -1);

    double sample_real_uniform(double a, double b);

    torch::Tensor sample_real_uniform(double a, double b, torch::IntArrayRef shape);

    torch::Tensor sample_real_uniform(double a, double b, const torch::Tensor& in);

    torch::Tensor sample_real_uniform(const torch::Tensor& a, const torch::Tensor& b);

    int64_t sample_int_uniform(int64_t a, int64_t b);

    torch::Tensor sample_int_uniform(int64_t a, int64_t b, torch::IntArrayRef shape);

    double sample_gaussian(double mu, double sigma);

    torch::Tensor sample_gaussian(double mu, double sigma, torch::IntArrayRef shape);

    torch::Tensor gen_int_permutation(int64_t n);
};     
}
}