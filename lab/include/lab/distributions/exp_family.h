#pragma once

#include "lab/distributions/base.h"

namespace lab
{
namespace distributions
{

class ExponentialFamily : public Distribution 
{
    LAB_ARG(torch::TensorList, natural_params);
    LAB_ARG(torch::Tensor , mean_carrier_measure);
public:
    using Distribution::Distribution;

    virtual torch::Tensor log_normalizer(torch::TensorList params);
    
    torch::Tensor entropy() override;
};

}

}

