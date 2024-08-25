#include "lab/utils/net.h"

namespace lab
{

namespace utils
{

bool is_torch_cuda_available()
{
    return torch::cuda::is_available();
}

torch::Device get_torch_device()
{
    return is_torch_cuda_available() ? torch::kCUDA : torch::kCPU;
}

torch::nn::Sequential build_fc_model(const std::vector<int64_t>& dims, const torch::nn::AnyModule& activation)
{
    LAB_CHECK_GE(dims.size(), 2);
    torch::nn::Sequential model;
    for(size_t i = 0; i < dims.size() - 1; i++)
    {
        auto linear = torch::nn::Linear(dims[i], dims[i + 1]);
        model->push_back(linear);
        auto act = activation.clone();
        model->push_back(act);
    }
    return model;
}

torch::nn::AnyModule get_act_fn(const std::string& name)
{
    if (name == "ReLU") return torch::nn::AnyModule(torch::nn::ReLU());
    else if (name == "LeakyReLU") return torch::nn::AnyModule(torch::nn::LeakyReLU());
    else if (name == "ELU") return torch::nn::AnyModule(torch::nn::ELU());
    else if (name == "SELU") return torch::nn::AnyModule(torch::nn::SELU());
    else if (name == "SiLU") return torch::nn::AnyModule(torch::nn::SiLU());
    else if (name == "Sigmoid") return torch::nn::AnyModule(torch::nn::Sigmoid());
    else if (name == "LogSigmoid") return torch::nn::AnyModule(torch::nn::LogSigmoid());
    else if (name == "Softmax") return torch::nn::AnyModule(torch::nn::Softmax(1));
    else if (name == "LogSoftmax") return torch::nn::AnyModule(torch::nn::LogSoftmax(1));
    else if (name == "Tanh") return torch::nn::AnyModule(torch::nn::Tanh());

    LAB_LOG_FATAL("Unsupported activation function!");
    return torch::nn::AnyModule();
}

torch::nn::AnyModule get_loss_fn(const std::string& name)
{
    if (name == "MSELoss") return torch::nn::AnyModule(torch::nn::MSELoss());
    else if (name == "CrossEntropyLoss") return torch::nn::AnyModule(torch::nn::CrossEntropyLoss());
    else if (name == "NLLLoss") return torch::nn::AnyModule(torch::nn::NLLLoss());
    else if (name == "BCELoss") return torch::nn::AnyModule(torch::nn::BCELoss());
    else if (name == "BCEWithLogitsLoss") return torch::nn::AnyModule(torch::nn::BCEWithLogitsLoss());

    LAB_LOG_FATAL("Unsupported loss function!");
    return torch::nn::AnyModule();
}

std::shared_ptr<torch::optim::Optimizer> get_optim(const std::shared_ptr<torch::nn::Module>& net, const OptimSpec& optim_spec)
{
    std::string name = optim_spec.name;
    
    //if (name == "SGD") return std::make_shared<torch::optim::SGD>(net.parameters());
    if (name == "Adam") return std::make_shared<torch::optim::Adam>(net->parameters());
    else if (name == "GlobalAdam") return std::make_shared<torch::optim::GlobalAdam>(net->parameters());
    else if (name == "RAdam") return std::make_shared<torch::optim::RAdam>(net->parameters());
    else if (name == "RMSprop") return std::make_shared<torch::optim::RMSprop>(net->parameters());
    else if (name == "GlobalRMSprop") return std::make_shared<torch::optim::GlobalRMSprop>(net->parameters());

    LAB_LOG_FATAL("Unsupported optimizer!");
    return nullptr;
}

std::shared_ptr<torch::optim::LRScheduler> get_lr_schedular(const std::shared_ptr<torch::optim::Optimizer>& optimizer, const LrSchedulerSpec& spec)
{
    std::string name = spec.name;
    if(name == "StepLR") return std::make_shared<torch::optim::StepLR>(*optimizer, spec.step_size, spec.gamma);

    LAB_LOG_FATAL("Unsupported Schedular!");
    return nullptr;
}

torch::nn::init::NonlinearityType get_nonlinearirty_type(const std::string& name)
{
    if (name == "Linear") return torch::enumtype::kLinear();
    else if (name == "Conv1D") return torch::enumtype::kConv1D();
    else if (name == "Conv2D") return torch::enumtype::kConv2D();
    else if (name == "Conv3D") return torch::enumtype::kConv3D();
    else if (name == "ConvTranspose1D") return torch::enumtype::kConvTranspose1D();
    else if (name == "ConvTranspose2D") return torch::enumtype::kConvTranspose2D();
    else if (name == "ConvTranspose3D") return torch::enumtype::kConvTranspose3D();
    else if (name == "Sigmoid") return torch::enumtype::kSigmoid();
    else if (name == "Tanh") return torch::enumtype::kTanh();
    else if (name == "ReLU") return torch::enumtype::kReLU();
    else if (name == "LeakyReLU") return torch::enumtype::kLeakyReLU();

    LAB_LOG_FATAL("Unsupported type!");
    return torch::nn::init::NonlinearityType(); // For quite warning
}

}

}