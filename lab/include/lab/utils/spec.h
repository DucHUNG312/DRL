#pragma once

#include "lab/common/common.h"
#include <nlohmann/json_fwd.hpp>
namespace lab
{

namespace utils
{

struct VarSchedulerSpec
{
    std::string updater;
    double start_val;
    double end_val;
    uint64_t start_step;
    uint64_t end_step;

    LAB_DEFAULT_CONSTRUCT(VarSchedulerSpec);
};
struct AlgorithmSpec
{
    VarSchedulerSpec explore_spec;
    VarSchedulerSpec entropy_spec;
    std::string name;
    std::string action_pdtype;
    std::string action_policy;
    double policy_loss_coef;
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
    LrSchedulerSpec lr_scheduler_spec;
    OptimSpec optim_spec;
    LossSpec loss_spec;
    std::vector<std::string> out_layers_activation;
    std::vector<int64_t> hid_layers;
    std::string name;
    std::string hid_layers_activation;
    std::string init_fn;
    //std::vector<int64_t> clip_grad_val;
    std::string update_type;
    double update_frequency;
    uint64_t polyak_coef;
    bool gpu;

    LAB_DEFAULT_CONSTRUCT(NetSpec);
};

struct AgentSpec
{
    std::string name;

    LAB_DEFAULT_CONSTRUCT(AgentSpec);
};

struct BodySpec
{
    AlgorithmSpec algorithm;
    NetSpec net;
    MemorySpec memory;
    std::string product;
    uint64_t num;

    LAB_DEFAULT_CONSTRUCT(BodySpec);
};

struct RendererSpec
{
    std::string graphics;
    uint64_t screen_width;
    uint64_t screen_height;
    bool enabled;

    LAB_DEFAULT_CONSTRUCT(RendererSpec);
};

struct EnvSpec
{
    RendererSpec renderer;
    std::vector<double> reward_range;
    std::string name;
    std::string frame_op;
    double reward_threshold;
    double reward_scale;
    int64_t max_frame;
    int64_t clock_speed;
    uint64_t frame_op_len;
    uint64_t max_time;
    uint64_t num_envs;
    uint64_t seed;
    bool nondeterministic;
    bool auto_reset;
    bool normalize_state;

    LAB_DEFAULT_CONSTRUCT(EnvSpec);
};

struct MetaSpec
{
    std::string search;
    uint64_t log_frequency;
    uint64_t eval_frequency;
    uint64_t max_session;
    uint64_t max_trial;
    bool distributed = false;
    bool resume = false;
    bool rigorous_eval = false;

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

class SpecLoader
{
public:
    SpecLoader() = default;
    virtual ~SpecLoader() = default;
    LAB_NONCOPYMOVEABLE(SpecLoader);

    static std::string get_default_experiment_name(const std::string& env_name);
    static EnvSpec load_default_env_spec(const std::string& env_name);
    static void load_specs_from_file(const std::string& file_path, const std::string& experiment_name);
private:
    static void load_spec_json(const nlohmann::json& j);
    static nlohmann::json get_json_stream(const std::string& file_path, const std::string& experiment_name);
    static VarSchedulerSpec get_explore_var_spec(const nlohmann::json& j);
    static VarSchedulerSpec get_entropy_coef_spec(const nlohmann::json& j);
    static AlgorithmSpec get_algorithm_spec(const nlohmann::json& j);
    static MemorySpec get_memory_spec(const nlohmann::json& j);
    static LossSpec get_loss_fn_spec(const nlohmann::json& j);
    static OptimSpec get_optim_spec(const nlohmann::json& j);
    static LrSchedulerSpec get_lr_scheduler_spec(const nlohmann::json& j);
    static NetSpec get_net_spec(const nlohmann::json& j);
    static AgentSpec get_agent_spec(const nlohmann::json& j);
    static RendererSpec get_renderer_spec(const nlohmann::json& j);
    static EnvSpec get_env_spec(const nlohmann::json& j, int64_t num = 0);
    static BodySpec get_body_spec(const nlohmann::json& j, int64_t num = 0);
    static MetaSpec get_meta_info_spec(const nlohmann::json& j);
public:
    static LabSpec specs;
    static std::unordered_map<std::string, std::string> default_env_spec_path;
};


}

}