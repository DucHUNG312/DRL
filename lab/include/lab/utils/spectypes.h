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

    ExploreVarSpec() = default;
    ExploreVarSpec(const ExploreVarSpec& other) = default;
    ExploreVarSpec& operator=(const ExploreVarSpec& other) = default;
    ExploreVarSpec(ExploreVarSpec&& other) noexcept = default;
    ExploreVarSpec& operator=(ExploreVarSpec&& other) noexcept = default;
    ~ExploreVarSpec() = default;
};

struct EntropyCoefSpec
{
    std::string name;
    double start_val;
    double end_val;
    uint64_t start_step;
    uint64_t end_step;

    EntropyCoefSpec() = default;
    EntropyCoefSpec(const EntropyCoefSpec& other) = default;
    EntropyCoefSpec& operator=(const EntropyCoefSpec& other) = default;
    EntropyCoefSpec(EntropyCoefSpec&& other) noexcept = default;
    EntropyCoefSpec& operator=(EntropyCoefSpec&& other) noexcept = default;
    ~EntropyCoefSpec() = default;
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

    AlgorithmSpec() = default;
    AlgorithmSpec(const AlgorithmSpec& other) = default;
    AlgorithmSpec& operator=(const AlgorithmSpec& other) = default;
    AlgorithmSpec(AlgorithmSpec&& other) noexcept = default;
    AlgorithmSpec& operator=(AlgorithmSpec&& other) noexcept = default;
    ~AlgorithmSpec() = default;
};

struct MemorySpec
{
    std::string name;
    uint64_t batch_size;
    uint64_t max_size;
    bool use_cer;

    MemorySpec() = default;
    MemorySpec(const MemorySpec& other) = default;
    MemorySpec& operator=(const MemorySpec& other) = default;
    MemorySpec(MemorySpec&& other) noexcept = default;
    MemorySpec& operator=(MemorySpec&& other) noexcept = default;
    ~MemorySpec() = default;
};

struct LossSpec
{
    std::string name;

    LossSpec() = default;
    LossSpec(const LossSpec& other) = default;
    LossSpec& operator=(const LossSpec& other) = default;
    LossSpec(LossSpec&& other) noexcept = default;
    LossSpec& operator=(LossSpec&& other) noexcept = default;
    ~LossSpec() = default;
};

struct OptimSpec
{
    std::string name;
    double lr;

    OptimSpec() = default;
    OptimSpec(const OptimSpec& other) = default;
    OptimSpec& operator=(const OptimSpec& other) = default;
    OptimSpec(OptimSpec&& other) noexcept = default;
    OptimSpec& operator=(OptimSpec&& other) noexcept = default;
    ~OptimSpec() = default;
};

struct LrSchedulerSpec
{
    std::string name;
    uint64_t step_size;
    double gamma;

    LrSchedulerSpec() = default;
    LrSchedulerSpec(const LrSchedulerSpec& other) = default;
    LrSchedulerSpec& operator=(const LrSchedulerSpec& other) = default;
    LrSchedulerSpec(LrSchedulerSpec&& other) noexcept = default;
    LrSchedulerSpec& operator=(LrSchedulerSpec&& other) noexcept = default;
    ~LrSchedulerSpec() = default;
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

    NetSpec() = default;
    NetSpec(const NetSpec& other) = default;
    NetSpec& operator=(const NetSpec& other) = default;
    NetSpec(NetSpec&& other) noexcept = default;
    NetSpec& operator=(NetSpec&& other) noexcept = default;
    ~NetSpec() = default;
};

struct AgentSpec
{
    std::string name;
    AlgorithmSpec algorithm;
    MemorySpec memory;
    NetSpec net;

    AgentSpec() = default;
    AgentSpec(const AgentSpec& other) = default;
    AgentSpec& operator=(const AgentSpec& other) = default;
    AgentSpec(AgentSpec&& other) noexcept = default;
    AgentSpec& operator=(AgentSpec&& other) noexcept = default;
    ~AgentSpec() = default;
};

struct RendererSpec
{
    bool enabled;
    std::string graphics;
    uint64_t screen_width;
    uint64_t screen_height;

    RendererSpec() = default;
    RendererSpec(const RendererSpec& other) = default;
    RendererSpec& operator=(const RendererSpec& other) = default;
    RendererSpec(RendererSpec&& other) noexcept = default;
    RendererSpec& operator=(RendererSpec&& other) noexcept = default;
    ~RendererSpec() = default;
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

    EnvSpec() = default;
    EnvSpec(const EnvSpec& other) = default;
    EnvSpec& operator=(const EnvSpec& other) = default;
    EnvSpec(EnvSpec&& other) noexcept = default;
    EnvSpec& operator=(EnvSpec&& other) noexcept = default;
    ~EnvSpec() = default;
};

struct BodySpec
{
    std::string product;
    uint64_t num;

    BodySpec() = default;
    BodySpec(const BodySpec& other) = default;
    BodySpec& operator=(const BodySpec& other) = default;
    BodySpec(BodySpec&& other) noexcept = default;
    BodySpec& operator=(BodySpec&& other) noexcept = default;
    ~BodySpec() = default;
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

    MetaSpec() = default;
    MetaSpec(const MetaSpec& other) = default;
    MetaSpec& operator=(const MetaSpec& other) = default;
    MetaSpec(MetaSpec&& other) noexcept = default;
    MetaSpec& operator=(MetaSpec&& other) noexcept = default;
    ~MetaSpec() = default;
};

struct LabSpec
{
    AgentSpec agent;
    EnvSpec env;
    BodySpec body;
    MetaSpec meta;

    LabSpec() = default;
    LabSpec(const LabSpec& other) = default;
    LabSpec& operator=(const LabSpec& other) = default;
    LabSpec(LabSpec&& other) noexcept = default;
    LabSpec& operator=(LabSpec&& other) noexcept = default;
    ~LabSpec() = default;
};

ActionPdType str_to_action_pd_type(const std::string& str);

ActionPolicyType str_to_action_policy_type(const std::string& str);

NetType str_to_net_type(const std::string& str);

UpdateType str_to_update_type(const std::string& str);

ActivationFnType str_to_activation_type(const std::string& str);

SearchType str_to_search_type(const std::string& str);

}

}