#include "lab/agents/net/mlp.h"

namespace lab
{

namespace agents
{


MLPNetImpl::MLPNetImpl(const utils::NetSpec& spec, int64_t in_dim, torch::IntArrayRef out_dim)
    : NetImpl(spec, in_dim, out_dim)
{
    spec_.hid_layers.insert(spec_.hid_layers.begin(), in_dim);

    hid_layers_activation_ = utils::create_act(spec_.hid_layers_activation);
    model_ = utils::create_fc_model(spec_.hid_layers, hid_layers_activation_);
    
    if(out_dim.size() > 1 && spec_.out_layers_activation.size() == 1)
        for(int64_t i = 0; i < out_dim.size() - 1; i++)
            spec_.out_layers_activation.push_back(spec_.out_layers_activation[0]);

    LAB_CHECK_EQ(out_dim.size(), spec_.out_layers_activation.size());
    out_layers_activations_.reserve(out_dim.size());

    for (int64_t i = 0; i < out_dim.size(); i++)
    { 
        std::vector<int64_t> out_dims = { spec_.hid_layers[spec_.hid_layers.size() - 1], out_dim[i] }; 
        out_layers_activations_.push_back(utils::create_act(spec_.out_layers_activation[i]));
        model_tail_->push_back(utils::create_fc_model(out_dims, out_layers_activations_[i]));
    }

    loss_function_ = utils::create_loss(spec_.loss_spec.name);

    model_ = register_module("model", model_);
    model_tail_ = register_module("model_tail", model_tail_);

    // transfer to device
    to(device_);
    to(torch::kDouble);

    // enable training mode
    train();
}

torch::Tensor MLPNetImpl::forward(torch::Tensor x)
{
    x = model_->forward(x);
    std::vector<torch::Tensor> outputs;
    for (const auto& module : *model_tail_)
    {
        auto seq_module = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(module);
        outputs.push_back(seq_module->forward(x));
    }
    return torch::stack(outputs).squeeze(-1);
}

}

}