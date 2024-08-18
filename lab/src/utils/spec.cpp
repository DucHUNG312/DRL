#include "lab/utils/spec.h"

namespace lab
{

namespace utils
{

LabSpec SpecLoader::specs;

std::unordered_map<std::string, std::string> SpecLoader::default_env_spec_path
{
    { "CartPole", join_paths(std::string(EXPERIMENT_SPEC_DIR), "reinforce/reinforce_cartpole.json") }
};

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
        spec.name = get_json_value<std::string>(j_explore["name"]);
        spec.start_val = get_json_value<double>(j_explore["start_val"]);
        spec.end_val = get_json_value<double>(j_explore["end_val"]);
        spec.start_step = get_json_value<uint64_t>(j_explore["start_step"]);
        spec.end_step = get_json_value<uint64_t>(j_explore["end_step"]);
    }
    return spec;
}

EntropyCoefSpec SpecLoader::get_entropy_coef_spec(const json& j) 
{
    EntropyCoefSpec spec;
    if(j.contains("entropy_coef_spec"))
    {
        json j_entropy = j["entropy_coef_spec"];
        spec.name = get_json_value<std::string>(j_entropy["name"]);
        spec.start_val = get_json_value<double>(j_entropy["start_val"]);
        spec.end_val = get_json_value<double>(j_entropy["end_val"]);
        spec.start_step = get_json_value<uint64_t>(j_entropy["start_step"]);
        spec.end_step = get_json_value<uint64_t>(j_entropy["end_step"]);
    }
    return spec;
}

AlgorithmSpec SpecLoader::get_algorithm_spec(const json& j) 
{
    AlgorithmSpec spec;
    if(j.contains("algorithm"))
    {
        json j_algo = j["algorithm"];
        spec.name = j_algo["name"];
        spec.action_pdtype = str_to_action_pd_type(get_json_value<std::string>(j_algo["action_pdtype"]));
        spec.action_policy = str_to_action_policy_type(get_json_value<std::string>(j_algo["action_policy"]));
        spec.explore_spec = get_explore_var_spec(j_algo);
        spec.entropy_spec = get_entropy_coef_spec(j_algo);
        spec.gamma = get_json_value<double>(j_algo["gamma"]);
        spec.training_frequency = get_json_value<uint64_t>(j_algo["training_frequency"]);
        spec.training_batch_iter = get_json_value<uint64_t>(j_algo["training_batch_iter"]);
        spec.training_iter = get_json_value<uint64_t>(j_algo["training_iter"]);
        spec.training_start_step = get_json_value<uint64_t>(j_algo["training_start_step"]);
    }
    return spec;
}

MemorySpec SpecLoader::get_memory_spec(const json& j) 
{
    MemorySpec spec;
    if(j.contains("memory"))
    {
        json j_mem = j["memory"];
        spec.name = get_json_value<std::string>(j_mem["name"]);
        spec.batch_size = get_json_value<uint64_t>(j_mem["batch_size"]);
        spec.max_size = get_json_value<uint64_t>(j_mem["max_size"]);
        spec.use_cer = get_json_value<bool>(j_mem["use_cer"]);
    }
    return spec;
}

LossSpec SpecLoader::get_loss_fn_spec(const json& j) 
{
    LossSpec spec;
    if(j.contains("loss_spec"))
    {
        json j_loss = j["loss_spec"];
        spec.name = get_json_value<std::string>(j_loss["name"]);
    }
    return spec;
}

OptimSpec SpecLoader::get_optim_spec(const json& j) 
{
    OptimSpec spec;
    if(j.contains("optim_spec"))
    {
        json j_optim = j["optim_spec"];
        spec.name = get_json_value<std::string>(j_optim["name"]);
        spec.lr = get_json_value<double>(j_optim["lr"]);
    }
    return spec;
}

LrSchedulerSpec SpecLoader::get_lr_scheduler_spec(const json& j) 
{
    LrSchedulerSpec spec;
    if(j.contains("lr_scheduler_spec"))
    {
        json j_lr_scheduler = j["lr_scheduler_spec"];
        spec.name = get_json_value<std::string>(j_lr_scheduler["name"]);
        spec.step_size = get_json_value<uint64_t>(j_lr_scheduler["step_size"]);
        spec.gamma = get_json_value<double>(j_lr_scheduler["gamma"]);
    }
    return spec;
}

NetSpec SpecLoader::get_net_spec(const json& j) 
{
    NetSpec spec;
    if(j.contains("net"))
    {
        json j_net = j["net"];
        spec.type = str_to_net_type(j_net["type"]);
        spec.hid_layers = get_json_value<std::vector<int64_t>>(j_net["hid_layers"]);
        spec.hid_layers_activation = get_json_value<std::string>(j_net["hid_layers_activation"]);
        spec.out_layers_activation = get_json_value<std::vector<std::string>>(j_net["out_layers_activation"]);
        spec.init_fn = get_json_value<std::string>(j_net["init_fn"]);
        //spec.clip_grad_val = get_json_value<std::vector<int64_t>>(j_net["clip_grad_val"]);
        spec.loss_spec = get_loss_fn_spec(j_net);
        spec.optim_spec = get_optim_spec(j_net);
        spec.lr_scheduler_spec = get_lr_scheduler_spec(j_net);
        spec.update_type = str_to_update_type(get_json_value<std::string>(j_net["update_type"]));
        spec.update_frequency = get_json_value<double>(j_net["update_frequency"]);
        spec.polyak_coef = get_json_value<uint64_t>(j_net["polyak_coef"]);
        spec.gpu = get_json_value<bool>(j_net["gpu"]);
    }
    return spec;
}

AgentSpec SpecLoader::get_agent_spec(const json& j, int64_t num /*= 0*/) 
{
    AgentSpec spec;
    if(j.contains("agent") && j["agent"].size() > num)
    {
        json j_agent = j["agent"][num];
        spec.name = get_json_value<std::string>(j_agent["name"]);
        spec.algorithm = get_algorithm_spec(j_agent);
        spec.memory = get_memory_spec(j_agent);
        spec.net = get_net_spec(j_agent);
    }
    return spec;
}

RendererSpec SpecLoader::get_renderer_spec(const json& j) 
{
    RendererSpec spec;
    if(j.contains("renderer"))
    {
        json j_render = j["renderer"];
        spec.enabled = get_json_value<bool>(j_render["enabled"]);
        spec.graphics = get_json_value<std::string>(j_render["graphics"]);
        spec.screen_width = get_json_value<uint64_t>(j_render["screen_width"]);
        spec.screen_height = get_json_value<uint64_t>(j_render["screen_height"]);
    }
    return spec;
}

EnvSpec SpecLoader::get_env_spec(const json& j, int64_t num /*= 0*/) 
{
    EnvSpec spec;
    if(j.contains("env") && j["env"].size() > num)
    {
        json j_env = j["env"][num];
        spec.name = get_json_value<std::string>(j_env["name"]);
        spec.frame_op = get_json_value<std::string>(j_env["frame_op"]);
        spec.frame_op_len = get_json_value<uint64_t>(j_env["frame_op_len"]);
        spec.max_time = get_json_value<uint64_t>(j_env["max_time"]);
        spec.max_frame = get_json_value<uint64_t>(j_env["max_frame"]);
        spec.reward_threshold = get_json_value<double>(j_env["reward_threshold"]);
        spec.reward_scale = get_json_value<double>(j_env["reward_scale"]);
        spec.reward_range = get_json_value<std::vector<double>>(j_env["reward_range"]);
        spec.seed = get_json_value<uint64_t>(j_env["seed"]);
        spec.nondeterministic = get_json_value<bool>(j_env["nondeterministic"]);
        spec.auto_reset = get_json_value<bool>(j_env["auto_reset"]);
        spec.normalize_state = get_json_value<bool>(j_env["normalize_state"]);
        spec.is_open = get_json_value<bool>(j_env["is_open"]);
        spec.renderer = get_renderer_spec(j_env);
    }
    return spec;
}

BodySpec SpecLoader::get_body_spec(const json& j) 
{
    BodySpec spec;
    if(j.contains("body"))
    {
        json j_body = j["body"];
        spec.product = get_json_value<std::string>(j_body["product"]);
        spec.num = get_json_value<uint64_t>(j_body["num"]);
    }
    return spec;
}

MetaSpec SpecLoader::get_meta_info_spec(const json& j) 
{
    MetaSpec spec;
    if(j.contains("meta"))
    {
        json j_meta = j["meta"];
        spec.distributed = get_json_value<bool>(j_meta["distributed"]);
        spec.resume = get_json_value<bool>(j_meta["resume"]);
        spec.rigorous_eval = get_json_value<bool>(j_meta["rigorous_eval"]);
        spec.log_frequency = get_json_value<uint64_t>(j_meta["log_frequency"]);
        spec.eval_frequency = get_json_value<uint64_t>(j_meta["eval_frequency"]);
        spec.max_session = get_json_value<uint64_t>(j_meta["max_session"]);
        spec.max_trial = get_json_value<uint64_t>(j_meta["max_trial"]);
        spec.search = str_to_search_type(get_json_value<std::string>(j_meta["search"]));
    }
    return spec;
}

std::string SpecLoader::get_default_experiment_name(const std::string& env_name)
{
    if(env_name == "CartPole") return "reinforce_cartpole";
    LAB_LOG_FATAL("Unknown environment!");
    return std::string();
}

EnvSpec SpecLoader::load_default_env_spec(const std::string& env_name)
{
    load_specs_from_file(default_env_spec_path[env_name], get_default_experiment_name(env_name));
    return specs.env;
}
    
}

}