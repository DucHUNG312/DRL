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

    Rand(const Rand& other);

    Rand(Rand&& other) noexcept;

    Rand& operator=(const Rand& other) 
    {
        if (this != &other) 
        {
            generator = other.generator;
            seed = other.seed;
        }
        return *this;
    }

    Rand& operator=(Rand&& other) noexcept 
    {
        if (this != &other) 
        {
            generator = std::move(other.generator);
            seed = std::move(other.seed);
        }
        return *this;
    }

    void set_seed(uint64_t seed);

    int64_t choice(const torch::Tensor& prob, double u = -1);

    double sample_real_uniform(double a, double b);

    torch::Tensor sample_real_uniform(double a, double b, const torch::IntArrayRef& shape);

    torch::Tensor sample_real_uniform(torch::Tensor& a, torch::Tensor& b);

    int64_t sample_int_uniform(int64_t a, int64_t b);

    torch::Tensor sample_int_uniform(int64_t a, int64_t b, const torch::IntArrayRef& shape);

    double sample_gaussian(double mu, double sigma);

    torch::Tensor sample_gaussian(double mu, double sigma, const torch::IntArrayRef& shape);

    torch::Tensor gen_int_permutation(int64_t n);
};     
}
}