#pragma once

#include "lab/core.h"

namespace lab
{
namespace utils
{
class Rand
{
public:
    static torch::Generator generator;
    static uint seed;
public:
    Rand(uint64_t seed = 42);
    ~Rand(){};

    static void set_seed(uint64_t seed);

    static int64_t choice(const torch::Tensor& prob, double u = -1);

    static double sample_real_uniform(double a, double b);

    static torch::Tensor sample_real_uniform(double a, double b, const torch::IntArrayRef& shape);

    static int64_t sample_int_uniform(int64_t a, int64_t b);

    static torch::Tensor sample_int_uniform(int64_t a, int64_t b, const torch::IntArrayRef& shape);

    static double sample_gaussian(double mu, double sigma);

    static torch::Tensor sample_gaussian(double mu, double sigma, const torch::IntArrayRef& shape);
};     
}
}