#pragma once

#include "lab/agents/algorithms/algorithms.h"
#include "lab/agents/memory/memory.h"
#include "lab/agents/net/net.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-requires"
#endif
#include <DataFrame/DataFrame.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif


namespace lab
{
namespace agents
{

template<typename ObsSpace, typename ActSpace>
class Body
{
public:
    using ObsType = typename ObsSpace::Type;
    using ActType = typename ActSpace::Type;
    using EnvType = envs::Env<ObsSpace, ActSpace>;
    using MemoryType = Memory<ObsSpace, ActSpace>;
    using AgentType = Agent<ObsSpace, ActSpace>;
    using TrainningDFType = hmdf::StdDataFrame<unsigned long>;

    LAB_ARG(EnvType, env);
    LAB_ARG(AgentType, agent);
    LAB_ARG(MemoryType, memory);
    LAB_ARG(utils::LabSpec, spec);
    LAB_ARG(int64_t, a);
    LAB_ARG(int64_t, e);
    LAB_ARG(int64_t, b);
    LAB_ARG(double, explore_var);
    LAB_ARG(double, entropy_coef);
    LAB_ARG(double, loss);
    LAB_ARG(double, mean_entropy);
    LAB_ARG(double, mean_grad_norm);
    LAB_ARG(double, best_total_reward_ma);
    LAB_ARG(double, total_reward_ma);
    LAB_ARG(TrainningDFType, train_df);
    LAB_ARG(ObsSpace, observation_space);
    LAB_ARG(ActSpace, action_space);
    LAB_ARG(int64_t, observable_dim);
    LAB_ARG(int64_t, action_dim);
    LAB_ARG(bool, is_discrete);
    LAB_ARG(ObsType, state);
    LAB_ARG(utils::ActionPdType, action_pdtype);
public:
    Body() = default;
    Body(const EnvType& env, const utils::LabSpec& spec, torch::IntArrayRef aeb = {0, 0, 0})
    {
        env_ = env;
        spec_ = spec; 
        a_ = aeb[0]; 
        e_ = aeb[1];
        b_ = aeb[2];
        
        train_df_.load_column<std::string>("epi", std::vector<std::string>{});
        train_df_.load_column<double>("t", std::vector<double>{});
        train_df_.load_column<double>("wall_t", std::vector<double>{});
        train_df_.load_column<int>("opt_step", std::vector<int>{});
        train_df_.load_column<int>("frame", std::vector<int>{});
        train_df_.load_column<double>("fps", std::vector<double>{});
        train_df_.load_column<double>("total_reward", std::vector<double>{});
        train_df_.load_column<double>("total_reward_ma", std::vector<double>{});
        train_df_.load_column<double>("loss", std::vector<double>{});
        train_df_.load_column<double>("lr", std::vector<double>{});
        train_df_.load_column<double>("explore_var", std::vector<double>{});
        train_df_.load_column<double>("entropy_coef", std::vector<double>{});
        train_df_.load_column<double>("entropy", std::vector<double>{});
        train_df_.load_column<double>("grad_norm", std::vector<double>{});

        observation_space_ = env_.observation_space();
        action_space_ = env_.action_space();
        observable_dim_ = env_.get_observable_dim();
        action_dim_ = env_.get_action_dim();
        is_discrete_ = env_.is_discrete();
        state_ = env_.result().state;

        action_pdtype_ = spec_.agent.algorithm.action_pdtype;
    }

    void calc_df_row()
    {

    }
};

template<typename ObsSpace, typename ActSpace>
class Agent
{
    using ObsType = typename ObsSpace::Type;
    using ActType = typename ActSpace::Type;
    using BodyType = Body<ObsSpace, ActSpace>;
    using AlgorithmType = Algorithm<ObsSpace, ActSpace>;
     
    LAB_ARG(utils::AgentSpec, spec);
    LAB_ARG(BodyType, body);
    LAB_ARG(AlgorithmType, algorithm);
public:
    Agent() = default;
    Agent(const BodyType& body)
    {
        body_ = body;
        spec(body_.spec().agent);
        body_.agent(*this);
        body_.memory(spec_.memory, body_);
        algorithm(AlgorithmType(*this));
    }
    virtual ~Agent() = default;

    ActType act(const ActType& action)
    {
        torch::NoGradGuard no_grad;
        return algorithm_.act(action);
    }
};

}
}