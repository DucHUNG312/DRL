#pragma once

#include "lab/core.h"
#include "lab/utils/file.h"
#include "lab/utils/spectypes.h"
#include "lab/utils/algorithm.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace lab
{

namespace utils
{

class SpecLoader
{
public:
    SpecLoader() = default;

    ~SpecLoader() = default;

    LAB_NONCOPYMOVEABLE(SpecLoader);

    static void load_spec_json(const json& j);

    static json get_json_stream(const std::string& file_path, const std::string& experiment_name);

    static void load_specs_from_file(const std::string& file_path, const std::string& experiment_name);

    static ExploreVarSpec get_explore_var_spec(const json& j);

    static EntropyCoefSpec get_entropy_coef_spec(const json& j);

    static AlgorithmSpec get_algorithm_spec(const json& j);

    static MemorySpec get_memory_spec(const json& j);

    static LossSpec get_loss_fn_spec(const json& j);

    static OptimSpec get_optim_spec(const json& j);

    static LrSchedulerSpec get_lr_scheduler_spec(const json& j);

    static NetSpec get_net_spec(const json& j);

    static AgentSpec get_agent_spec(const json& j, int64_t num = 0);

    static RendererSpec get_renderer_spec(const json& j);

    static EnvSpec get_env_spec(const json& j, int64_t num = 0);

    static BodySpec get_body_spec(const json& j);

    static MetaSpec get_meta_info_spec(const json& j);

    static std::string get_default_experiment_name(const std::string& env_name);

    static EnvSpec load_default_env_spec(const std::string& env_name); 
public:
    static LabSpec specs;
    static std::unordered_map<std::string, std::string> default_env_spec_path;
};

template<typename T>
LAB_FORCE_INLINE T get_json_value(const json& j)
{
    return j.is_null() ? T() : j.get<T>();
}


}

}