#include "main.h"

using namespace lab;

int main(int argc, char** argv)
{
    LAB_INIT_LOG();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}