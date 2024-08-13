#include "lab/register.h"

TORCH_LIBRARY(lab, m) 
{
    m.class_<lab::spaces::Discrete>("Discrete")
        .def(torch::init<int64_t, int64_t>())
        .def("sample", &lab::spaces::Discrete::sample)
        .def("contains", &lab::spaces::Discrete::contains)
        .def("n", [](const c10::intrusive_ptr<lab::spaces::Discrete>& self){
          return self->n();
        })
        .def("start", [](const c10::intrusive_ptr<lab::spaces::Discrete>& self){
          return self->start();
        })
        .def("shape", [](const c10::intrusive_ptr<lab::spaces::Discrete>& self){
          return self->shape().to_torch();
        });

    m.class_<lab::spaces::Box>("Box")
        .def(torch::init<const torch::Tensor&, const torch::Tensor&>())
        .def("sample", &lab::spaces::Box::sample)
        .def("contains", &lab::spaces::Box::contains)
        .def("low", [](const c10::intrusive_ptr<lab::spaces::Box>& self){
          return self->low();
        })
        .def("high", [](const c10::intrusive_ptr<lab::spaces::Box>& self){
          return self->high();
        })
        .def("shape", [](const c10::intrusive_ptr<lab::spaces::Box>& self){
          return self->shape().to_torch();
        });
}
