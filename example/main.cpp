#include <lab/lab.h>

int main(int argc, char** argv)
{
    LAB_INIT_LOG();

    torch::Tensor prob = torch::tensor({0.2, 0.5, 0.3}, torch::kDouble);
    int64_t choice = lab::utils::Rand::choice(prob);

    LAB_LOG_DEBUG("{}", choice);

    LAB_SHUTDOWN_LOG();

    return 0;
}