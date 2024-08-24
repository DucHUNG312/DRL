#include "lab/agents/net/mlp.h"

namespace lab
{

namespace agents
{


MLPNetImpl::MLPNetImpl(const utils::NetSpec& spec, int64_t in_dim, torch::IntArrayRef out_dim)
    : Net(spec, in_dim, out_dim)
{
    spec_.hid_layers.insert(spec_.hid_layers.begin(), in_dim);
    model_ = utils::build_fc_model(spec_.hid_layers, utils::get_act_fn(spec_.hid_layers_activation));
    if(out_dim.size() > 1 && spec_.out_layers_activation.size() == 1)
        for(int64_t i = 0; i < out_dim.size() - 1; i++)
            spec_.out_layers_activation.push_back(spec_.out_layers_activation[0]);

    LAB_CHECK_EQ(out_dim.size(), spec_.out_layers_activation.size());

    for (int64_t i = 0; i < out_dim.size(); i++)
    { 
        std::vector<int64_t> out_dims = { spec_.hid_layers[spec_.hid_layers.size() - 1], out_dim[i] }; 
        model_tail_->push_back(utils::build_fc_model(out_dims, utils::get_act_fn(spec_.out_layers_activation[i])));
    }

    loss_fn_ = utils::get_loss_fn(spec_.loss_spec.name);

    this->to(device_);
    this->train();
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
    return torch::cat(outputs, 1);
}

}

}