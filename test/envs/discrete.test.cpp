#include "main.h"

TEST(Spaces, Discrete1)
{
    lab::spaces::Discrete space(5);

    ASSERT_EQ(space.name(), lab::spaces::SpaceType::DISCRETE);
    ASSERT_EQ(space.n(), 5);
    ASSERT_EQ(space.shape().size(), 1);
    ASSERT_EQ(space.shape(), torch::IntArrayRef(1));
    ASSERT_EQ(space.rand().seed, 0);
    ASSERT_EQ(space.rand().generator.current_seed(), 0);
    ASSERT_TRUE(space.contains(0));
    ASSERT_TRUE(space.contains(4));
    ASSERT_TRUE(!space.contains(5));

    bool ok = true;
    for(int i = 0; i<15; i++)
    {
        ok = ok && space.contains(space.sample());
    }
    ASSERT_TRUE(ok);
}

TEST(Spaces, Discrete2)
{
    lab::spaces::Discrete space(5, 3);
    ASSERT_EQ(space.name(), lab::spaces::SpaceType::DISCRETE);
    ASSERT_EQ(space.n(), 5);
    ASSERT_EQ(space.shape().size(), 1);
    ASSERT_EQ(space.shape(), torch::IntArrayRef(1));
    ASSERT_EQ(space.rand().seed, 0);
    ASSERT_EQ(space.rand().generator.current_seed(), 0);
    ASSERT_TRUE(space.contains(3));
    ASSERT_TRUE(space.contains(7));
    ASSERT_TRUE(!space.contains(8));

    bool ok = true;
    for(int i = 0; i<15; i++)
    {
        ok = ok && space.contains(space.sample());
    }
    ASSERT_TRUE(ok);
}

TEST(Spaces, Discrete3)
{
    lab::spaces::Discrete space(5, 3);
    space.set_seed(42);
    ASSERT_EQ(space.name(), lab::spaces::SpaceType::DISCRETE);
    ASSERT_EQ(space.n(), 5 );
    ASSERT_EQ(space.shape().size(), 1 );
    ASSERT_EQ(space.shape(), torch::IntArrayRef(1) );
    ASSERT_EQ(space.rand().seed, 42 );
    ASSERT_EQ(space.rand().generator.current_seed(), 42 );
    ASSERT_TRUE( space.contains(3) );
    ASSERT_TRUE( space.contains(7) );
    ASSERT_TRUE( !space.contains(8) );

    bool ok = true;
    for(int i = 0; i<15; i++)
    {
        ok = ok && space.contains(space.sample());
    }
    ASSERT_TRUE(ok);
}

TEST(Spaces, Discrete4)
{
    lab::spaces::Discrete space1(5, 3);
    lab::spaces::Discrete space2(space1);
    ASSERT_EQ( space2.name(), lab::spaces::SpaceType::DISCRETE );
    ASSERT_EQ( space2.n(), 5 );
    ASSERT_EQ( space2.shape().size(), 1 );
    ASSERT_EQ( space2.shape(), torch::IntArrayRef(1) );
    ASSERT_EQ( space2.rand().seed, 0 );
    ASSERT_EQ( space2.rand().generator.current_seed(), 0 );
    ASSERT_TRUE( space2.contains(3) );
    ASSERT_TRUE( space2.contains(7) );
    ASSERT_TRUE( !space2.contains(8) );

    bool ok = true;
    for(int i = 0; i<15; i++)
    {
        ok = ok && space2.contains(space2.sample());
    }
    ASSERT_TRUE(ok);

    ASSERT_EQ( space1, space2 );
}


TEST(Spaces, Discrete5)
{
    lab::spaces::Discrete space1(5, 0);
    lab::spaces::Discrete space2 = lab::spaces::make_discrete_space(5);
    lab::spaces::Discrete space3 = lab::spaces::make_discrete_space(6);

    ASSERT_EQ( space2.name(), lab::spaces::SpaceType::DISCRETE );
    ASSERT_EQ( space2.n(), 5 );
    ASSERT_EQ( space2.shape().size(), 1 );
    ASSERT_EQ( space2.shape(), torch::IntArrayRef(1) );
    ASSERT_EQ( space2.rand().seed, 0 );
    ASSERT_EQ( space2.rand().generator.current_seed(), 0 );
    ASSERT_TRUE( space2.contains(0) );
    ASSERT_TRUE( space2.contains(3) );
    ASSERT_TRUE( !space2.contains(5) );

    ASSERT_EQ( space3.name(), lab::spaces::SpaceType::DISCRETE );
    ASSERT_EQ( space3.n(), 6 );
    ASSERT_EQ( space3.shape().size(), 1 );
    ASSERT_EQ( space3.shape(), torch::IntArrayRef(1) );
    ASSERT_EQ( space3.rand().seed, 0 );
    ASSERT_EQ( space3.rand().generator.current_seed(), 0 );
    ASSERT_TRUE( space3.contains(0) );
    ASSERT_TRUE( space3.contains(3) );
    ASSERT_TRUE( !space3.contains(6) );

    ASSERT_EQ( space1, space2 );
    ASSERT_TRUE( space1 != space3 );
    ASSERT_TRUE( space2 != space3 );
}

TEST(Spaces, Discrete6)
{
    lab::spaces::Discrete space1(5, 0);
    lab::spaces::Discrete space2 = lab::spaces::make_discrete_space(5, 0);
    lab::spaces::Discrete space3 = lab::spaces::make_discrete_space(6, 5);

    ASSERT_EQ( space2.name(), lab::spaces::SpaceType::DISCRETE );
    ASSERT_EQ( space2.n(), 5 );
    ASSERT_EQ( space2.shape().size(), 1 );
    ASSERT_EQ( space2.shape(), torch::IntArrayRef(1) );
    ASSERT_EQ( space2.rand().seed, 0 );
    ASSERT_EQ( space2.rand().generator.current_seed(), 0 );
    ASSERT_TRUE( space2.contains(0) );
    ASSERT_TRUE( space2.contains(3) );
    ASSERT_TRUE( !space2.contains(5) );

    ASSERT_EQ( space3.name(), lab::spaces::SpaceType::DISCRETE );
    ASSERT_EQ( space3.n(), 6 );
    ASSERT_EQ( space3.shape().size(), 1 );
    ASSERT_EQ( space3.shape(), torch::IntArrayRef(1) );
    ASSERT_EQ( space3.rand().seed, 0 );
    ASSERT_EQ( space3.rand().generator.current_seed(), 0 );
    ASSERT_TRUE( space3.contains(5) );
    ASSERT_TRUE( space3.contains(8) );
    ASSERT_TRUE( !space3.contains(11) );

    ASSERT_EQ( space1, space2 );
    ASSERT_TRUE( space1 != space3 );
    ASSERT_TRUE( space2 != space3 );
}

TEST(Spaces, Discrete7)
{
    lab::spaces::Discrete space1(5, 0);
    space1.set_seed(42);
    lab::spaces::Discrete space2 = space1;
    lab::spaces::Discrete space3;
    space3 = space1;

    ASSERT_EQ( space2.name(), lab::spaces::SpaceType::DISCRETE );
    ASSERT_EQ( space2.n(), 5 );
    ASSERT_EQ( space2.shape().size(), 1 );
    ASSERT_EQ( space2.shape(), torch::IntArrayRef(1) );
    ASSERT_EQ( space2.rand().seed, 42 );
    ASSERT_EQ( space2.rand().generator.current_seed(), 42 );
    ASSERT_TRUE( space2.contains(0) );
    ASSERT_TRUE( space2.contains(3) );
    ASSERT_TRUE( !space2.contains(5) );

    ASSERT_EQ( space3.name(), lab::spaces::SpaceType::DISCRETE );
    ASSERT_EQ( space3.n(), 5 );
    ASSERT_EQ( space3.shape().size(), 1 );
    ASSERT_EQ( space3.shape(), torch::IntArrayRef(1) );
    ASSERT_EQ( space3.rand().seed, 42 );
    ASSERT_EQ( space3.rand().generator.current_seed(), 42 );
    ASSERT_TRUE( space3.contains(0) );
    ASSERT_TRUE( space3.contains(3) );
    ASSERT_TRUE( !space3.contains(5) );

    ASSERT_EQ( space1, space2 );
    ASSERT_EQ( space1, space3 );
    ASSERT_EQ( space2, space3 );
}

