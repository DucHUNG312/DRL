#include "lab/utils/spectypes.h"

namespace lab
{

namespace utils
{

ActionPdType str_to_action_pd_type(const std::string& str)
{
    if (str == "Categorical")
        return ActionPdType::CATEGORICAL;
    else if(str == "Argmax")
        return ActionPdType::ARGMAX;
    return ActionPdType::ACTION_PD_NONE;
}

ActionPolicyType str_to_action_policy_type(const std::string& str)
{
    if (str == "epsilon_greedy")
        return ActionPolicyType::EPSILON_GREEDY;
    else if(str == "boltzmann")
        return ActionPolicyType::BOLTZMANN;
    return ActionPolicyType::ACTION_POLICY_NONE;
}

NetType str_to_net_type(const std::string& str)
{
    if (str == "MLPNet")
        return NetType::MLPNET;
    return NetType::NET_NONE;
}

UpdateType str_to_update_type(const std::string& str)
{
    if (str == "polyak")
        return UpdateType::POLYAK;
    return UpdateType::UPDATE_NONE;
}

ActivationFnType str_to_activation_type(const std::string& str)
{
    if (str == "relu")
        return ActivationFnType::RELU;
    else if (str == "leaky_relu")
        return ActivationFnType::LEAKY_RELU;
    else if (str == "selu")
        return ActivationFnType::SELU;
    else if (str == "silu")
        return ActivationFnType::SILU;
    else if (str == "sigmoid")
        return ActivationFnType::SIGMOID;
    else if (str == "log_sigmoid")
        return ActivationFnType::LOG_SIGMOID;
    else if (str == "tanh")
        return ActivationFnType::TANH;
    return ActivationFnType::ACTIVATION_FN_NONE;
}

SearchType str_to_search_type(const std::string& str)
{
    if (str == "RandomSearch")
        return SearchType::RANDOM;
    return SearchType::SEARCH_NONE;
}

}

}