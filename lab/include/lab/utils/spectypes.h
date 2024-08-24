#pragma once

#include "lab/core.h"

namespace lab
{

namespace utils
{

enum ActionPdType
{
    ACTION_PD_NONE,
    CATEGORICAL,
    ARGMAX
};

enum ActionPolicyType
{
    ACTION_POLICY_NONE,
    EPSILON_GREEDY,
    BOLTZMANN
};

enum NetType
{
    NET_NONE,
    MLPNET
};

enum UpdateType
{
    UPDATE_NONE,
    POLYAK
};

enum ActivationFnType
{
    ACTIVATION_FN_NONE,
    RELU,
    LEAKY_RELU,
    SELU,
    SILU,
    SIGMOID,
    LOG_SIGMOID,
    TANH
};

enum SearchType
{
    SEARCH_NONE,
    RANDOM
};;

struct ExploreVarSpec
{
    std::string name;
    double start_val;
    double end_val;
    uint64_t start_step;
    uint64_t end_step;

    LAB_DEFAULT_CONSTRUCT(ExploreVarSpec);
};

struct EntropyCoefSpec
{
    std::string name;
    double start_val;
    double end_val;
    uint64_t start_step;
    uint64_t end_step;

    LAB_DEFAULT_CONSTRUCT(EntropyCoefSpec);
};

struct AlgorithmSpec
{
    std::string name;
    ActionPdType action_pdtype;
    ActionPolicyType action_policy;
    ExploreVarSpec explore_spec;
    EntropyCoefSpec entropy_spec;
    double gamma;
    uint64_t training_frequency;
    uint64_t training_batch_iter;
    uint64_t training_iter;
    uint64_t training_start_step;

    LAB_DEFAULT_CONSTRUCT(AlgorithmSpec);
};

struct MemorySpec
{
    std::string name;
    uint64_t batch_size;
    uint64_t max_size;
    bool use_cer;

    LAB_DEFAULT_CONSTRUCT(MemorySpec);
};

struct LossSpec
{
    std::string name;

    LAB_DEFAULT_CONSTRUCT(LossSpec);
};

struct OptimSpec
{
    std::string name;
    double lr;

    LAB_DEFAULT_CONSTRUCT(OptimSpec);
};

struct LrSchedulerSpec
{
    std::string name;
    uint64_t step_size;
    double gamma;

    LAB_DEFAULT_CONSTRUCT(LrSchedulerSpec);
};

struct NetSpec
{
    NetType type;
    std::vector<int64_t> hid_layers;
    std::string hid_layers_activation;
    std::vector<std::string> out_layers_activation;
    std::string init_fn;
    //std::vector<int64_t> clip_grad_val;
    LossSpec loss_spec;
    OptimSpec optim_spec;
    LrSchedulerSpec lr_scheduler_spec;
    UpdateType update_type;
    double update_frequency;
    uint64_t polyak_coef;
    bool gpu;

    LAB_DEFAULT_CONSTRUCT(NetSpec);
};

struct AgentSpec
{
    std::string name;
    AlgorithmSpec algorithm;
    MemorySpec memory;
    NetSpec net;

    LAB_DEFAULT_CONSTRUCT(AgentSpec);
};

struct RendererSpec
{
    bool enabled;
    std::string graphics;
    uint64_t screen_width;
    uint64_t screen_height;

    LAB_DEFAULT_CONSTRUCT(RendererSpec);
};

struct EnvSpec
{
    std::string name;
    std::string frame_op;
    uint64_t frame_op_len;
    uint64_t max_time;
    uint64_t max_frame;
    double reward_threshold;
    double reward_scale;
    std::vector<double> reward_range;
    uint64_t seed;
    bool nondeterministic;
    bool auto_reset;
    bool normalize_state;
    bool is_open;
    RendererSpec renderer;

    LAB_DEFAULT_CONSTRUCT(EnvSpec);
};

struct BodySpec
{
    std::string product;
    uint64_t num;

    LAB_DEFAULT_CONSTRUCT(BodySpec);
};

struct MetaSpec
{
    bool distributed = false;
    bool resume = false;
    bool rigorous_eval = false;
    uint64_t log_frequency;
    uint64_t eval_frequency;
    uint64_t max_session;
    uint64_t max_trial;
    SearchType search;

    LAB_DEFAULT_CONSTRUCT(MetaSpec);
};

struct LabSpec
{
    AgentSpec agent;
    EnvSpec env;
    BodySpec body;
    MetaSpec meta;

    LAB_DEFAULT_CONSTRUCT(LabSpec);
};

ActionPdType str_to_action_pd_type(const std::string& str);

ActionPolicyType str_to_action_policy_type(const std::string& str);

NetType str_to_net_type(const std::string& str);

UpdateType str_to_update_type(const std::string& str);

ActivationFnType str_to_activation_type(const std::string& str);

SearchType str_to_search_type(const std::string& str);

}

}