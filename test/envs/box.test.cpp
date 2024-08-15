#include "main.h"

TEST(Spaces, Box1)
{
    torch::Tensor low = torch::full({1}, 0, torch::TensorOptions().dtype(torch::kDouble));
    torch::Tensor high = torch::full({1}, 10, torch::TensorOptions().dtype(torch::kDouble));
    torch::Tensor low3 = torch::full({1}, 5, torch::TensorOptions().dtype(torch::kDouble));
    torch::Tensor high3 = torch::full({1}, 15, torch::TensorOptions().dtype(torch::kDouble));
    torch::Tensor valid = torch::full({1}, 5, torch::TensorOptions().dtype(torch::kDouble));
    torch::Tensor not_valid_1 = torch::full({1}, 12, torch::TensorOptions().dtype(torch::kDouble));
    torch::Tensor not_valid_2 = torch::full({1}, 16, torch::TensorOptions().dtype(torch::kDouble));
    lab::spaces::Box space1(low, high);
    lab::spaces::Box space2(space1);
    lab::spaces::Box space3(low3, high3);

    ASSERT_EQ( space1.name(), lab::spaces::SpaceType::BOX );
    ASSERT_TRUE( lab::utils::tensor_eq(space1.low(), torch::full({1}, 0)) );
    ASSERT_TRUE( lab::utils::tensor_eq(space1.high(), torch::full({1}, 10)) );
    ASSERT_EQ( space1.shape(), low.sizes() );
    ASSERT_EQ( space1.shape(), high.sizes() );
    ASSERT_EQ( space1.shape().size(), low.dim() );
    ASSERT_EQ( space1.shape().size(), high.dim() );
    ASSERT_EQ( space1.shape().size(), 1 );
    ASSERT_EQ( space1.shape()[0], 1 );
    ASSERT_EQ( space1.rand().seed, 0 );
    ASSERT_EQ( space1.rand().generator.current_seed(), 0 );
    ASSERT_TRUE( space1.contains(valid) );
    ASSERT_TRUE( !space1.contains(not_valid_1) );
    ASSERT_TRUE( !space1.contains(not_valid_2) );

    ASSERT_EQ( space3.name(), lab::spaces::SpaceType::BOX );
    ASSERT_TRUE( lab::utils::tensor_eq(space3.low(), torch::full({1}, 5)) );
    ASSERT_TRUE( lab::utils::tensor_eq(space3.high(), torch::full({1}, 15)) );
    ASSERT_EQ( space3.shape(), low3.sizes() );
    ASSERT_EQ( space3.shape(), high3.sizes() );
    ASSERT_EQ( space3.shape().size(), low3.dim() );
    ASSERT_EQ( space3.shape().size(), high3.dim() );
    ASSERT_EQ( space3.shape().size(), 1 );
    ASSERT_EQ( space3.shape()[0], 1 );
    ASSERT_EQ( space3.rand().seed, 0 );
    ASSERT_EQ( space3.rand().generator.current_seed(), 0 );
    ASSERT_TRUE( space3.contains(valid) );
    ASSERT_TRUE( !space3.contains(not_valid_2) );

    ASSERT_EQ( space1, space2 );
    ASSERT_TRUE( space1 != space3 );
    ASSERT_TRUE( space2 != space3 );

    bool ok = true;
    for(int i = 0; i<25; i++)
    {
        ok = ok && space1.contains(space1.sample());
    }
    ASSERT_TRUE( ok );
}

TEST(Spaces, Box2)
{
    std::vector<double> low_val = {2, 4, 5, -1, 5};
    std::vector<double> high_val = {6, 9, 12, 3, 10};
    std::vector<double> low3_val = {4, 7, 10, 0, 8};
    std::vector<double> high3_val = {8, 14, 24, 8, 30};
    std::vector<double> valid_val = {3, 6, 6, -1, 9};
    std::vector<double> not_valid_1_val = {2, 6, 13, -2, 8};
    std::vector<double> not_valid_2_val = {-23, 12, 12, -1, 25};
    torch::Tensor low = torch::tensor(low_val, torch::kDouble);
    torch::Tensor high = torch::tensor(high_val, torch::kDouble);
    torch::Tensor low3 = torch::tensor(low3_val, torch::kDouble);
    torch::Tensor high3 = torch::tensor(high3_val, torch::kDouble);
    torch::Tensor valid = torch::tensor(valid_val, torch::kDouble);
    torch::Tensor not_valid_1 = torch::tensor(not_valid_1_val, torch::kDouble);
    torch::Tensor not_valid_2 = torch::tensor(not_valid_2_val, torch::kDouble);
    lab::spaces::Box space1(low, high);
    lab::spaces::Box space2 = space1;
    lab::spaces::Box space3(low3, high3);

    ASSERT_EQ( space1.name(), lab::spaces::SpaceType::BOX );
    ASSERT_TRUE( lab::utils::tensor_eq(space1.low(), low) );
    ASSERT_TRUE( lab::utils::tensor_eq(space1.high(), high) );
    ASSERT_EQ( space1.shape(), low.sizes() );
    ASSERT_EQ( space1.shape(), high.sizes() );
    ASSERT_EQ( space1.shape().size(), low.dim() );
    ASSERT_EQ( space1.shape().size(), high.dim() );
    ASSERT_EQ( space1.shape().size(), 1 );
    ASSERT_EQ( space1.shape()[0], 5 );
    ASSERT_EQ( space1.rand().seed, 0 );
    ASSERT_EQ( space1.rand().generator.current_seed(), 0 );
    ASSERT_TRUE( space1.contains(valid) );
    ASSERT_TRUE( !space1.contains(not_valid_1) );
    ASSERT_TRUE( !space1.contains(not_valid_2) );

    ASSERT_EQ( space3.name(), lab::spaces::SpaceType::BOX );
    ASSERT_TRUE( lab::utils::tensor_eq(space3.low(), low3) );
    ASSERT_TRUE( lab::utils::tensor_eq(space3.high(), high3) );
    ASSERT_EQ( space3.shape(), low3.sizes() );
    ASSERT_EQ( space3.shape(), high3.sizes() );
    ASSERT_EQ( space3.shape().size(), low3.dim() );
    ASSERT_EQ( space3.shape().size(), high3.dim() );
    ASSERT_EQ( space3.shape().size(), 1 );
    ASSERT_EQ( space3.shape()[0], 5 );
    ASSERT_EQ( space3.rand().seed, 0 );
    ASSERT_EQ( space3.rand().generator.current_seed(), 0 );

    ASSERT_EQ( space1, space2 );
    ASSERT_TRUE( space1 != space3 );
    ASSERT_TRUE( space2 != space3 );

    bool ok = true;
    for(int i = 0; i<25; i++)
    {
        ok = ok && space1.contains(space1.sample());
    }
    ASSERT_TRUE( ok );
}

TEST(Spaces, Box3)
{
    torch::Tensor low = torch::tensor({{{2, 4}, {5, -1}}, {{5, 6}, {12, 6}}}, torch::kDouble);
    torch::Tensor high = torch::tensor({{{6, 9}, {12, 3}}, {{10, 23}, {35, 89}}}, torch::kDouble);
    torch::Tensor low3 = torch::tensor({{{4, 6}, {8, 1}}, {{7, 18}, {24, 56}}}, torch::kDouble);
    torch::Tensor high3 = torch::tensor({{{12, 19}, {412, 43}}, {{14, 45}, {350, 189}}}, torch::kDouble);
    torch::Tensor valid = torch::tensor({{{3, 6}, {6, -1}}, {{9, 12}, {26, 67}}}, torch::kDouble);
    torch::Tensor not_valid_1 = torch::tensor({{{4, 9}, {13, 23}}, {{11, 2}, {-35, 38}}}, torch::kDouble);
    torch::Tensor not_valid_2 = torch::tensor({{{6, 9}, {-12, 24}}, {{-104, 7}, {20, 8}}}, torch::kDouble);
    lab::spaces::Box space1(low, high);
    lab::spaces::Box space2 = space1;
    lab::spaces::Box space3(low3, high3);

    ASSERT_EQ( space1.name(), lab::spaces::SpaceType::BOX );
    ASSERT_TRUE( lab::utils::tensor_eq(space1.low(), low) );
    ASSERT_TRUE( lab::utils::tensor_eq(space1.high(), high) );
    ASSERT_EQ( space1.shape(), low.sizes() );
    ASSERT_EQ( space1.shape(), high.sizes() );
    ASSERT_EQ( space1.shape().size(), low.dim() );
    ASSERT_EQ( space1.shape().size(), high.dim() );
    ASSERT_EQ( space1.shape().size(), 3 );
    ASSERT_EQ( space1.shape()[0], 2 );
    ASSERT_EQ( space1.shape()[1], 2 );
    ASSERT_EQ( space1.shape()[2], 2 );
    ASSERT_EQ( space1.rand().seed, 0 );
    ASSERT_EQ( space1.rand().generator.current_seed(), 0 );
    ASSERT_TRUE( space1.contains(valid) );
    ASSERT_TRUE( !space1.contains(not_valid_1) );
    ASSERT_TRUE( !space1.contains(not_valid_2) );

    ASSERT_EQ( space3.name(), lab::spaces::SpaceType::BOX );
    ASSERT_TRUE( lab::utils::tensor_eq(space3.low(), low3) );
    ASSERT_TRUE( lab::utils::tensor_eq(space3.high(), high3) );
    ASSERT_EQ( space3.shape(), low3.sizes() );
    ASSERT_EQ( space3.shape(), high3.sizes() );
    ASSERT_EQ( space3.shape().size(), low3.dim() );
    ASSERT_EQ( space3.shape().size(), high3.dim() );
    ASSERT_EQ( space3.shape().size(), 3 );
    ASSERT_EQ( space3.shape()[0], 2 );
    ASSERT_EQ( space3.shape()[1], 2 );
    ASSERT_EQ( space3.shape()[2], 2 );
    ASSERT_EQ( space3.rand().seed, 0 );
    ASSERT_EQ( space3.rand().generator.current_seed(), 0 );

    ASSERT_EQ( space1, space2 );
    ASSERT_TRUE( space1 != space3 );
    ASSERT_TRUE( space2 != space3 );

    bool ok = true;
    for(int i = 0; i<25; i++)
    {
        ok = ok && space1.contains(space1.sample());
    }
    ASSERT_TRUE( ok );
}

TEST(Spaces, Box4)
{
    torch::Tensor low = torch::tensor({{{2, 4}, {5, -1}}, {{5, 6}, {12, 6}}}, torch::kDouble);
    torch::Tensor high = torch::tensor({{{6, 9}, {12, 3}}, {{10, 23}, {35, 89}}}, torch::kDouble);
    torch::Tensor low3 = torch::tensor({{{4, 6}, {8, 1}}, {{7, 18}, {24, 56}}}, torch::kDouble);
    torch::Tensor high3 = torch::tensor({{{12, 19}, {412, 43}}, {{14, 45}, {350, 189}}}, torch::kDouble);
    torch::Tensor valid = torch::tensor({{{3, 6}, {6, -1}}, {{9, 12}, {26, 67}}}, torch::kDouble);
    torch::Tensor not_valid_1 = torch::tensor({{{4, 9}, {13, 23}}, {{11, 2}, {-35, 38}}}, torch::kDouble);
    torch::Tensor not_valid_2 = torch::tensor({{{6, 9}, {-12, 24}}, {{-104, 7}, {20, 8}}}, torch::kDouble);
    lab::spaces::Box space1(low, high);
    lab::spaces::Box space2;
    space2 = space1;
    lab::spaces::Box space3(low3, high3);

    space1.set_seed(42);
    space3.set_seed(42);

    ASSERT_EQ( space1.name(), lab::spaces::SpaceType::BOX );
    ASSERT_TRUE( lab::utils::tensor_eq(space1.low(), low) );
    ASSERT_TRUE( lab::utils::tensor_eq(space1.high(), high) );
    ASSERT_EQ( space1.shape(), low.sizes() );
    ASSERT_EQ( space1.shape(), high.sizes() );
    ASSERT_EQ( space1.shape().size(), low.dim() );
    ASSERT_EQ( space1.shape().size(), high.dim() );
    ASSERT_EQ( space1.shape().size(), 3 );
    ASSERT_EQ( space1.shape()[0], 2 );
    ASSERT_EQ( space1.shape()[1], 2 );
    ASSERT_EQ( space1.shape()[2], 2 );
    ASSERT_EQ( space1.rand().seed, 42 );
    ASSERT_EQ( space1.rand().generator.current_seed(), 42 );
    ASSERT_TRUE( space1.contains(valid) );
    ASSERT_TRUE( !space1.contains(not_valid_1) );
    ASSERT_TRUE( !space1.contains(not_valid_2) );

    ASSERT_EQ( space3.name(), lab::spaces::SpaceType::BOX );
    ASSERT_TRUE( lab::utils::tensor_eq(space3.low(), low3) );
    ASSERT_TRUE( lab::utils::tensor_eq(space3.high(), high3) );
    ASSERT_EQ( space3.shape(), low3.sizes() );
    ASSERT_EQ( space3.shape(), high3.sizes() );
    ASSERT_EQ( space3.shape().size(), low3.dim() );
    ASSERT_EQ( space3.shape().size(), high3.dim() );
    ASSERT_EQ( space3.shape().size(), 3 );
    ASSERT_EQ( space3.shape()[0], 2 );
    ASSERT_EQ( space3.shape()[1], 2 );
    ASSERT_EQ( space3.shape()[2], 2 );
    ASSERT_EQ( space3.rand().seed, 42 );
    ASSERT_EQ( space3.rand().generator.current_seed(), 42 );

    ASSERT_EQ( space1, space2 );
    ASSERT_TRUE( space1 != space3 );
    ASSERT_TRUE( space2 != space3 );

    bool ok = true;
    for(int i = 0; i<25; i++)
    {
        ok = ok && space1.contains(space1.sample());
    }
    ASSERT_TRUE( ok );
}


TEST(Spaces, Box5)
{
    torch::Tensor low = torch::full({1}, 0, torch::kDouble);
    torch::Tensor high = torch::full({1}, 1, torch::kDouble);
    torch::Tensor valid = torch::full({1}, 0.5, torch::kDouble);
    torch::Tensor not_valid_2 = torch::full({1}, 1.5, torch::kDouble);
    lab::spaces::Box space1 = lab::spaces::make_box_space();
    lab::spaces::Box space2(space1);

    ASSERT_EQ( space1.name(), lab::spaces::SpaceType::BOX );
    ASSERT_TRUE( lab::utils::tensor_eq(space1.low(), low) );
    ASSERT_TRUE( lab::utils::tensor_eq(space1.high(), high) );
    ASSERT_EQ( space1.shape(), low.sizes() );
    ASSERT_EQ( space1.shape(), high.sizes() );
    ASSERT_EQ( space1.shape().size(), low.dim() );
    ASSERT_EQ( space1.shape().size(), high.dim() );
    ASSERT_EQ( space1.shape().size(), 1 );
    ASSERT_EQ( space1.shape()[0], 1 );
    ASSERT_EQ( space1.rand().seed, 0 );
    ASSERT_EQ( space1.rand().generator.current_seed(), 0 );
    ASSERT_TRUE( space1.contains(valid) );
    ASSERT_TRUE( !space1.contains(not_valid_2) );

    ASSERT_EQ( space1, space2 );

    bool ok = true;
    for(int i = 0; i<25; i++)
    {
        ok = ok && space1.contains(space1.sample());
    }
    ASSERT_TRUE( ok );
}

TEST(Spaces, Box6)
{
    torch::Tensor low = torch::full({2, 2, 2} , 0, torch::kDouble);
    torch::Tensor high = torch::full({2, 2, 2}, 1, torch::kDouble);
    torch::Tensor low3 = torch::full({3, 3, 3} , 0, torch::kDouble);
    torch::Tensor high3 = torch::full({3, 3, 3}, 1, torch::kDouble);
    torch::Tensor valid = torch::full({2, 2, 2}, 0.5, torch::kDouble);
    torch::Tensor not_valid_1 = torch::full({2, 2, 2}, 2.5, torch::kDouble);
    torch::Tensor not_valid_2 = torch::full({2, 2, 2}, 1.5, torch::kDouble);
    lab::spaces::Box space1 = lab::spaces::make_box_space(torch::IntArrayRef({2, 2, 2}));
    lab::spaces::Box space2(space1);
    lab::spaces::Box space3 = lab::spaces::make_box_space(torch::IntArrayRef({3, 3, 3}));

    ASSERT_EQ( space1.name(), lab::spaces::SpaceType::BOX );
    ASSERT_TRUE( lab::utils::tensor_eq(space1.low(), low) );
    ASSERT_TRUE( lab::utils::tensor_eq(space1.high(), high) );
    ASSERT_EQ( space1.shape(), low.sizes() );
    ASSERT_EQ( space1.shape(), high.sizes() );
    ASSERT_EQ( space1.shape().size(), low.dim() );
    ASSERT_EQ( space1.shape().size(), high.dim() );
    ASSERT_EQ( space1.shape().size(), 3 );
    ASSERT_EQ( space1.shape()[0], 2 );
    ASSERT_EQ( space1.shape()[1], 2 );
    ASSERT_EQ( space1.shape()[2], 2 );
    ASSERT_EQ( space1.rand().seed, 0 );
    ASSERT_EQ( space1.rand().generator.current_seed(), 0 );
    ASSERT_TRUE( space1.contains(valid) );
    ASSERT_TRUE( !space1.contains(not_valid_1) );
    ASSERT_TRUE( !space1.contains(not_valid_2) );

    ASSERT_EQ( space3.name(), lab::spaces::SpaceType::BOX );
    ASSERT_TRUE( lab::utils::tensor_eq(space3.low(), low3) );
    ASSERT_TRUE( lab::utils::tensor_eq(space3.high(), high3) );
    ASSERT_EQ( space3.shape(), low3.sizes() );
    ASSERT_EQ( space3.shape(), high3.sizes() );
    ASSERT_EQ( space3.shape().size(), low3.dim() );
    ASSERT_EQ( space3.shape().size(), high3.dim() );
    ASSERT_EQ( space3.shape().size(), 3 );
    ASSERT_EQ( space3.shape()[0], 3 );
    ASSERT_EQ( space3.shape()[1], 3 );
    ASSERT_EQ( space3.shape()[2], 3 );
    ASSERT_EQ( space3.rand().seed, 0 );
    ASSERT_EQ( space3.rand().generator.current_seed(), 0 );

    ASSERT_EQ( space1, space2 );
    ASSERT_TRUE( space1 != space3 );
    ASSERT_TRUE( space2 != space3 );

    bool ok = true;
    for(int i = 0; i<25; i++)
    {
        ok = ok && space1.contains(space1.sample());
    }
    ASSERT_TRUE( ok );
}

TEST(Spaces, Box7)
{
    torch::Tensor low = torch::tensor({{{2, 4}, {5, -1}}, {{5, 6}, {12, 6}}}, torch::kDouble);
    torch::Tensor high = torch::tensor({{{6, 9}, {12, 3}}, {{10, 23}, {35, 89}}}, torch::kDouble);
    torch::Tensor low3 = torch::tensor({{{4, 6}, {8, 1}}, {{7, 18}, {24, 56}}}, torch::kDouble);
    torch::Tensor high3 = torch::tensor({{{12, 19}, {412, 43}}, {{14, 45}, {350, 189}}}, torch::kDouble);
    torch::Tensor valid = torch::tensor({{{3, 6}, {6, -1}}, {{9, 12}, {26, 67}}}, torch::kDouble);
    torch::Tensor not_valid_1 = torch::tensor({{{4, 9}, {13, 23}}, {{11, 2}, {-35, 38}}}, torch::kDouble);
    torch::Tensor not_valid_2 = torch::tensor({{{6, 9}, {-12, 24}}, {{-104, 7}, {20, 8}}}, torch::kDouble);
    lab::spaces::Box space1 = lab::spaces::make_box_space(low, high);
    lab::spaces::Box space2(space1);
    lab::spaces::Box space3 = lab::spaces::make_box_space(low3, high3);

    ASSERT_EQ( space1.name(), lab::spaces::SpaceType::BOX );
    ASSERT_TRUE( lab::utils::tensor_eq(space1.low(), low) );
    ASSERT_TRUE( lab::utils::tensor_eq(space1.high(), high) );
    ASSERT_EQ( space1.shape(), low.sizes() );
    ASSERT_EQ( space1.shape(), high.sizes() );
    ASSERT_EQ( space1.shape().size(), low.dim() );
    ASSERT_EQ( space1.shape().size(), high.dim() );
    ASSERT_EQ( space1.shape().size(), 3 );
    ASSERT_EQ( space1.shape()[0], 2 );
    ASSERT_EQ( space1.shape()[1], 2 );
    ASSERT_EQ( space1.shape()[2], 2 );
    ASSERT_EQ( space1.rand().seed, 0 );
    ASSERT_EQ( space1.rand().generator.current_seed(), 0 );
    ASSERT_TRUE( space1.contains(valid) );
    ASSERT_TRUE( !space1.contains(not_valid_1) );
    ASSERT_TRUE( !space1.contains(not_valid_2) );

    ASSERT_EQ( space3.name(), lab::spaces::SpaceType::BOX );
    ASSERT_TRUE( lab::utils::tensor_eq(space3.low(), low3) );
    ASSERT_TRUE( lab::utils::tensor_eq(space3.high(), high3) );
    ASSERT_EQ( space3.shape(), low3.sizes() );
    ASSERT_EQ( space3.shape(), high3.sizes() );
    ASSERT_EQ( space3.shape().size(), low3.dim() );
    ASSERT_EQ( space3.shape().size(), high3.dim() );
    ASSERT_EQ( space3.shape().size(), 3 );
    ASSERT_EQ( space3.shape()[0], 2 );
    ASSERT_EQ( space3.shape()[1], 2 );
    ASSERT_EQ( space3.shape()[2], 2 );
    ASSERT_EQ( space3.rand().seed, 0 );
    ASSERT_EQ( space3.rand().generator.current_seed(), 0 );

    ASSERT_EQ( space1, space2 );
    ASSERT_TRUE( space1 != space3 );
    ASSERT_TRUE( space2 != space3 );

    bool ok = true;
    for(int i = 0; i<25; i++)
    {
        ok = ok && space1.contains(space1.sample());
    }
    ASSERT_TRUE( ok );
}

TEST(Spaces, Box8)
{
    torch::Tensor low = torch::tensor({{{2, 4}, {5, -1}}, {{5, 6}, {12, 6}}}, torch::kDouble);
    torch::Tensor high = torch::tensor({{{6, 9}, {12, 3}}, {{10, 23}, {35, 89}}}, torch::kDouble);
    torch::Tensor low3 = torch::tensor({{{4, 6}, {8, 1}}, {{7, 18}, {24, 56}}}, torch::kDouble);
    torch::Tensor high3 = torch::tensor({{{12, 19}, {412, 43}}, {{14, 45}, {350, 189}}}, torch::kDouble);
    torch::Tensor valid = torch::tensor({{{3, 6}, {6, -1}}, {{9, 12}, {26, 67}}}, torch::kDouble);
    torch::Tensor not_valid_1 = torch::tensor({{{4, 9}, {13, 23}}, {{11, 2}, {-35, 38}}}, torch::kDouble);
    torch::Tensor not_valid_2 = torch::tensor({{{6, 9}, {-12, 24}}, {{-104, 7}, {20, 8}}}, torch::kDouble);
    lab::spaces::Box space1 = lab::spaces::make_box_space(low, high);
    lab::spaces::Box space2 = space1;
    lab::spaces::Box space3 = lab::spaces::make_box_space(low3, high3);

    ASSERT_EQ( space1.name(), lab::spaces::SpaceType::BOX );
    ASSERT_TRUE( lab::utils::tensor_eq(space1.low(), low) );
    ASSERT_TRUE( lab::utils::tensor_eq(space1.high(), high) );
    ASSERT_EQ( space1.shape(), low.sizes() );
    ASSERT_EQ( space1.shape(), high.sizes() );
    ASSERT_EQ( space1.shape().size(), low.dim() );
    ASSERT_EQ( space1.shape().size(), high.dim() );
    ASSERT_EQ( space1.shape().size(), 3 );
    ASSERT_EQ( space1.shape()[0], 2 );
    ASSERT_EQ( space1.shape()[1], 2 );
    ASSERT_EQ( space1.shape()[2], 2 );
    ASSERT_EQ( space1.rand().seed, 0 );
    ASSERT_EQ( space1.rand().generator.current_seed(), 0 );
    ASSERT_TRUE( space1.contains(valid) );
    ASSERT_TRUE( !space1.contains(not_valid_1) );
    ASSERT_TRUE( !space1.contains(not_valid_2) );

    ASSERT_EQ( space3.name(), lab::spaces::SpaceType::BOX );
    ASSERT_TRUE( lab::utils::tensor_eq(space3.low(), low3) );
    ASSERT_TRUE( lab::utils::tensor_eq(space3.high(), high3) );
    ASSERT_EQ( space3.shape(), low3.sizes() );
    ASSERT_EQ( space3.shape(), high3.sizes() );
    ASSERT_EQ( space3.shape().size(), low3.dim() );
    ASSERT_EQ( space3.shape().size(), high3.dim() );
    ASSERT_EQ( space3.shape().size(), 3 );
    ASSERT_EQ( space3.shape()[0], 2 );
    ASSERT_EQ( space3.shape()[1], 2 );
    ASSERT_EQ( space3.shape()[2], 2 );
    ASSERT_EQ( space3.rand().seed, 0 );
    ASSERT_EQ( space3.rand().generator.current_seed(), 0 );

    ASSERT_EQ( space1, space2 );
    ASSERT_TRUE( space1 != space3 );
    ASSERT_TRUE( space2 != space3 );

    bool ok = true;
    for(int i = 0; i<25; i++)
    {
        ok = ok && space1.contains(space1.sample());
    }
    ASSERT_TRUE( ok );
}