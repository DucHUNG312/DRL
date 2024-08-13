#include <lab/lab.h>

int main(int argc, char** argv)
{
    LAB_INIT_LOG();

    LAB_LOG_DEBUG("Discrete");

    lab::spaces::Discrete discrete_space(5);
    lab::spaces::Discrete discrete_space2 = lab::spaces::make_discrete_space(5);

    LAB_LOG_DEBUG(discrete_space.shape());
    LAB_LOG_DEBUG(discrete_space2.shape());

    LAB_LOG_DEBUG("Box");

    torch::Tensor tensor = torch::full({3, 3, 3} , 0, torch::kDouble);
    lab::spaces::Box box_space_2 = lab::spaces::make_box_space();
    lab::spaces::Box box_space = lab::spaces::make_box_space(torch::IntArrayRef({3, 3, 3}));
    
    LAB_LOG_DEBUG(tensor.sizes());
    LAB_LOG_DEBUG(box_space.shape());
    LAB_LOG_DEBUG(box_space_2.shape());

    bool ok = true;
    for(int i = 0; i<50; i++)
    {
        ok = ok && box_space.contains(box_space.sample());
        LAB_LOG_DEBUG(ok);
    }

    LAB_SHUTDOWN_LOG();

    return 0;
}