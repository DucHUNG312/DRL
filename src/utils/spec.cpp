#include "lab/utils/spec.h"

namespace lab
{

namespace utils
{

LabSpec SpecLoader::specs;

void SpecLoader::load_spec_json(const json& j)
{
    try
    {
        specs.agent = get_agent_spec(j);
        specs.env = get_env_spec(j);
        specs.body = get_body_spec(j);
        specs.meta = get_meta_info_spec(j);
    }
    catch(std::exception& e)
    {
        LAB_LOG_FATAL("{} at: {}({})", e.what(), __FILE__, __LINE__);
    }
}

json SpecLoader::get_json_stream(const std::string& file_path, const std::string& experiment_name) 
{
    std::ifstream json_file(file_path);
    if (!json_file.is_open()) 
    {
        LAB_LOG_FATAL("Cannot open file {}:{}({})", file_path, __FILE__, __LINE__);
    }
    json j;
    json_file >> j;
    return j[experiment_name];
}

void SpecLoader::load_specs_from_file(const std::string& file_path, const std::string& experiment_name)
{
    json j_lab = get_json_stream(file_path, experiment_name);
    load_spec_json(j_lab);
}

ExploreVarSpec SpecLoader::get_explore_var_spec(const json& j) 
{
    ExploreVarSpec spec;
    if(j.contains("explore_var_spec"))
    {
        json j_explore = j["explore_var_spec"];
        spec.name = j_explore["name"].get<std::string>();
        spec.start_val = j_explore["start_val"].get<double>();
        spec.end_val = j_explore["end_val"].get<double>();
        spec.start_step = j_explore["start_step"].get<uint64_t>();
        spec.end_step = j_explore["end_step"].get<uint64_t>();
    }
    return spec;
}

EntropyCoefSpec SpecLoader::get_entropy_coef_spec(const json& j) 
{
    EntropyCoefSpec spec;
    if(j.contains("entropy_coef_spec"))
    {
        json j_entropy = j["entropy_coef_spec"];
        spec.name = j_entropy["name"].get<std::string>();
        spec.start_val = j_entropy["start_val"].get<double>();
        spec.end_val = j_entropy["end_val"].get<double>();
        spec.start_step = j_entropy["start_step"].get<uint64_t>();
        spec.end_step = j_entropy["end_step"].get<uint64_t>();
    }
    return spec;
}

AlgorithmSpec SpecLoader::get_algorithm_spec(const json& j) 
{
    AlgorithmSpec spec;
    json j_algo = j["algorithm"];
    spec.name = j_algo["name"].get<std::string>();
    spec.action_pdtype = str_to_action_pd_type(j_algo["action_pdtype"].get<std::string>());
    spec.action_policy = str_to_action_policy_type(j_algo["action_policy"].get<std::string>());
    spec.explore_spec = get_explore_var_spec(j_algo);
    spec.entropy_spec = get_entropy_coef_spec(j_algo);
    spec.gamma = j_algo["gamma"].get<double>();
    spec.training_frequency = j_algo["training_frequency"].get<uint64_t>();
    spec.training_batch_iter = j_algo["training_batch_iter"].get<uint64_t>();
    spec.training_iter = j_algo["training_iter"].get<uint64_t>();
    spec.training_start_step = j_algo["training_start_step"].get<uint64_t>();
    return spec;
}

MemorySpec SpecLoader::get_memory_spec(const json& j) 
{
    MemorySpec spec;
    json j_mem = j["memory"];
    spec.name = j_mem["name"].get<std::string>();
    spec.batch_size = j_mem["batch_size"].get<uint64_t>();
    spec.max_size = j_mem["max_size"].get<uint64_t>();
    spec.use_cer = j_mem["use_cer"].get<bool>();
    return spec;
}

LossSpec SpecLoader::get_loss_fn_spec(const json& j) 
{
    LossSpec spec;
    json j_loss = j["loss_spec"];
    spec.name = j_loss["name"].get<std::string>();
    return spec;
}

OptimSpec SpecLoader::get_optim_spec(const json& j) 
{
    OptimSpec spec;
    json j_optim = j["optim_spec"];
    spec.name = j_optim["name"].get<std::string>();
    spec.lr = j_optim["lr"].get<double>();
    return spec;
}

LrSchedulerSpec SpecLoader::get_lr_scheduler_spec(const json& j) 
{
    LrSchedulerSpec spec;
    json j_lr_scheduler = j["lr_scheduler_spec"];
    spec.name = j_lr_scheduler["name"].get<std::string>();
    spec.step_size = j_lr_scheduler["step_size"].get<uint64_t>();
    spec.gamma = j_lr_scheduler["gamma"].get<double>();
    return spec;
}

NetSpec SpecLoader::get_net_spec(const json& j) 
{
    NetSpec spec;
    json j_net = j["net"];
    spec.type = str_to_net_type(j_net["type"].get<std::string>());
    spec.hid_layers = torch::IntArrayRef(j_net["hid_layers"].get<std::vector<int64_t>>());
    spec.hid_layers_activation = j_net["hid_layers_activation"].get<std::vector<std::string>>();
    spec.clip_grad_val = torch::IntArrayRef(j_net["clip_grad_val"].get<std::vector<int64_t>>());
    spec.loss_spec = get_loss_fn_spec(j_net);
    spec.optim_spec = get_optim_spec(j_net);
    spec.lr_scheduler_spec = get_lr_scheduler_spec(j_net);
    spec.update_type = str_to_update_type(j_net["update_type"]);
    spec.update_frequency = j_net["update_frequency"].get<double>();
    spec.polyak_coef = j_net.contains("polyak_coef") ? j_net["polyak_coef"].get<double>() : 0;
    spec.gpu = j_net["gpu"].get<bool>();
    return spec;
}

AgentSpec SpecLoader::get_agent_spec(const json& j, int64_t num /*= 0*/) 
{
    AgentSpec spec;
    json j_agent = j["agent"][num];
    spec.name = j_agent["name"].get<std::string>();
    spec.algorithm = get_algorithm_spec(j_agent);
    spec.memory = get_memory_spec(j_agent);
    spec.net = get_net_spec(j_agent);
    return spec;
}

EnvSpec SpecLoader::get_env_spec(const json& j, int64_t num /*= 0*/) 
{
    EnvSpec spec;
    json j_env = j["env"][num];
    spec.name = j_env["name"].get<std::string>();
    spec.frame_op = j_env["frame_op"].get<std::string>();
    spec.frame_op_len = j_env["frame_op_len"].get<uint64_t>();
    spec.max_time = j_env["max_time"].get<uint64_t>();
    spec.max_frame = j_env["max_frame"].get<uint64_t>();
    spec.normalize_state = j_env["normalize_state"].get<bool>();
    return spec;
}

BodySpec SpecLoader::get_body_spec(const json& j) 
{
    BodySpec spec;
    json j_body = j["body"];
    spec.product = j_body["product"].get<std::string>();
    spec.num = j_body["num"].get<uint64_t>();
    return spec;
}

MetaSpec SpecLoader::get_meta_info_spec(const json& j) 
{
    MetaSpec spec;
    json j_meta = j["meta"];
    spec.distributed = j_meta["distributed"].get<bool>();
    spec.resume = j_meta["resume"].get<bool>();
    spec.rigorous_eval = j_meta["rigorous_eval"].get<bool>();
    spec.log_frequency = j_meta["log_frequency"].get<uint64_t>();
    spec.eval_frequency = j_meta["eval_frequency"].get<uint64_t>();
    spec.max_session = j_meta["max_session"].get<uint64_t>();
    spec.max_trial = j_meta["max_trial"].get<uint64_t>();
    spec.search = str_to_search_type(j_meta["search"].get<std::string>());
    return spec;
}
    
}

}