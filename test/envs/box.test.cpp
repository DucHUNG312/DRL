#include "main.h"

TEST(Spaces, Box1)
{
    torch::Tensor low = torch::full({1}, 0, torch::TensorOptions().dtype(torch::kDouble));
    torch::Tensor high = torch::full({1}, 10, torch::TensorOptions().dtype(torch::kDouble));
    torch::Tensor low2 = torch::full({1}, 5, torch::TensorOptions().dtype(torch::kDouble));
    torch::Tensor high2 = torch::full({1}, 15, torch::TensorOptions().dtype(torch::kDouble));
    torch::Tensor valid = torch::full({1}, 5, torch::TensorOptions().dtype(torch::kDouble));
    torch::Tensor not_valid_1 = torch::full({1}, 12, torch::TensorOptions().dtype(torch::kDouble));
    torch::Tensor not_valid_2 = torch::full({1}, 16, torch::TensorOptions().dtype(torch::kDouble));
    lab::spaces::Box space1(low, high);
    lab::spaces::Box space2(low2, high2);

    ASSERT_EQ( space1->name(), "Box" );
    ASSERT_TRUE( lab::utils::tensor_eq(space1->low, torch::full({1}, 0)) );
    ASSERT_TRUE( lab::utils::tensor_eq(space1->high, torch::full({1}, 10)) );
    ASSERT_EQ( space1->shape().size(0), low.dim() );
    ASSERT_EQ( space1->shape().size(0), high.dim() );
    ASSERT_EQ( space1->shape().numel(), 1 );
    ASSERT_TRUE( space1->contains(valid) );
    ASSERT_TRUE( !space1->contains(not_valid_1) );
    ASSERT_TRUE( !space1->contains(not_valid_2) );

    ASSERT_EQ( space2->name(), "Box" );
    ASSERT_TRUE( lab::utils::tensor_eq(space2->low, torch::full({1}, 5)) );
    ASSERT_TRUE( lab::utils::tensor_eq(space2->high, torch::full({1}, 15)) );
    ASSERT_EQ( space2->shape().size(0), low2.dim() );
    ASSERT_EQ( space2->shape().size(0), high2.dim() );
    ASSERT_EQ( space2->shape().numel(), 1 );
    ASSERT_TRUE( space2->contains(valid) );
    ASSERT_TRUE( !space2->contains(not_valid_2) );

    bool ok = true;
    for(int i = 0; i<25; i++)
    {
        ok = ok && space1->contains(space1->sample());
    }
    ASSERT_TRUE( ok );
}

TEST(Spaces, Box2)
{
    std::vector<double> low_val = {2, 4, 5, -1, 5};
    std::vector<double> high_val = {6, 9, 12, 3, 10};
    std::vector<double> low2_val = {4, 7, 10, 0, 8};
    std::vector<double> high2_val = {8, 14, 24, 8, 30};
    std::vector<double> valid_val = {3, 6, 6, -1, 9};
    std::vector<double> not_valid_1_val = {2, 6, 13, -2, 8};
    std::vector<double> not_valid_2_val = {-23, 12, 12, -1, 25};
    torch::Tensor low = torch::tensor(low_val, torch::kDouble);
    torch::Tensor high = torch::tensor(high_val, torch::kDouble);
    torch::Tensor low2 = torch::tensor(low2_val, torch::kDouble);
    torch::Tensor high2 = torch::tensor(high2_val, torch::kDouble);
    torch::Tensor valid = torch::tensor(valid_val, torch::kDouble);
    torch::Tensor not_valid_1 = torch::tensor(not_valid_1_val, torch::kDouble);
    torch::Tensor not_valid_2 = torch::tensor(not_valid_2_val, torch::kDouble);
    lab::spaces::Box space1(low, high);
    lab::spaces::Box space2(low2, high2);

    ASSERT_EQ( space1->name(), "Box" );
    ASSERT_TRUE( lab::utils::tensor_eq(space1->low, low) );
    ASSERT_TRUE( lab::utils::tensor_eq(space1->high, high) );
    ASSERT_EQ( space1->shape().size(0), low.dim() );
    ASSERT_EQ( space1->shape().size(0), high.dim() );
    ASSERT_EQ( space1->shape()[0].item<double>(), 5 );
    ASSERT_EQ( space2->name(), "Box" );
    ASSERT_TRUE( lab::utils::tensor_eq(space2->low, low2) );
    ASSERT_TRUE( lab::utils::tensor_eq(space2->high, high2) );
    ASSERT_EQ( space2->shape().size(0), low2.dim() );
    ASSERT_EQ( space2->shape().size(0), high2.dim() );
    ASSERT_EQ( space2->shape()[0].item<double>(), 5 );
    ASSERT_TRUE( space1->contains(valid) );
    ASSERT_TRUE( !space1->contains(not_valid_1) );
    ASSERT_TRUE( !space1->contains(not_valid_2) );

    bool ok = true;
    for(int i = 0; i<25; i++)
    {
        ok = ok && space1->contains(space1->sample());
    }
    ASSERT_TRUE( ok );
}

TEST(Spaces, Box3)
{
    torch::Tensor low = torch::tensor({{{2, 4}, {5, -1}}, {{5, 6}, {12, 6}}}, torch::kDouble);
    torch::Tensor high = torch::tensor({{{6, 9}, {12, 3}}, {{10, 23}, {35, 89}}}, torch::kDouble);
    torch::Tensor low2 = torch::tensor({{{4, 6}, {8, 1}}, {{7, 18}, {24, 56}}}, torch::kDouble);
    torch::Tensor high2 = torch::tensor({{{12, 19}, {412, 43}}, {{14, 45}, {350, 189}}}, torch::kDouble);
    torch::Tensor valid = torch::tensor({{{3, 6}, {6, -1}}, {{9, 12}, {26, 67}}}, torch::kDouble);
    torch::Tensor not_valid_1 = torch::tensor({{{4, 9}, {13, 23}}, {{11, 2}, {-35, 38}}}, torch::kDouble);
    torch::Tensor not_valid_2 = torch::tensor({{{6, 9}, {-12, 24}}, {{-104, 7}, {20, 8}}}, torch::kDouble);
    lab::spaces::Box space1 = lab::spaces::make_box_space(low, high);
    lab::spaces::Box space2(space1);
    lab::spaces::Box space3 = space2;

    ASSERT_EQ( space3->name(), "Box" );
    ASSERT_TRUE( lab::utils::tensor_eq(space3->low, low) );
    ASSERT_TRUE( lab::utils::tensor_eq(space3->high, high) );
    ASSERT_EQ( space3->shape().size(0), low.dim() );
    ASSERT_EQ( space3->shape().size(0), high.dim() );
    ASSERT_EQ( space3->shape()[0].item<double>(), 2 );
    ASSERT_EQ( space3->shape()[1].item<double>(), 2 );
    ASSERT_EQ( space3->shape()[2].item<double>(), 2 );
    ASSERT_TRUE( space3->contains(valid) );
    ASSERT_TRUE( !space3->contains(not_valid_1) );
    ASSERT_TRUE( !space3->contains(not_valid_2) );

    bool ok = true;
    for(int i = 0; i<25; i++)
    {
        ok = ok && space3->contains(space3->sample());
    }
    ASSERT_TRUE( ok );
}