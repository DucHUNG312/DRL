#pragma once

#include "lab/core.h"
#include "lab/envs/base.h"
#include "lab/utils/spec.h"
#include "lab/agents/memory/base.h"
#include "lab/agents/algorithms/base.h"

namespace lab
{
namespace agents
{

class Algorithm;

class Body : public std::enable_shared_from_this<Body>
{
    LAB_ARG(std::shared_ptr<envs::Env>, env);
    LAB_ARG(std::shared_ptr<Memory>, memory);
    LAB_ARG(std::shared_ptr<Algorithm>, algorithm);
    LAB_ARG(torch::Tensor, loss);
    LAB_ARG(utils::BodySpec, spec);
public:
    Body(const std::shared_ptr<envs::Env>& env, const utils::BodySpec& spec);
    LAB_DEFAULT_CONSTRUCT(Body);

    void update(const envs::StepResult& result)
    {
        memory_->update(std::move(result));
    }

    torch::IValue act(const torch::Tensor& state)
    {
        torch::IValue action = algorithm_->act(std::move(state));
        return action;
    }

    torch::Tensor train() 
    {
        return algorithm_->train();
    }

    void save(torch::serialize::OutputArchive& archive) const
    {

    }

    void load(torch::serialize::InputArchive& archive)
    {

    }

    std::shared_ptr<spaces::AnySpace>& get_action_spaces();    
};

LAB_FORCE_INLINE torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Body>& body)
{
    return archive;
}

LAB_FORCE_INLINE torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Body>& body)
{
    return archive;
}


}

}