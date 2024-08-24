#include "main.h"

TEST(Spaces, Discrete1)
{
    lab::spaces::Discrete space(5);

    ASSERT_EQ(space->name_, "Discrete");
    ASSERT_EQ(space->options.n(), 5);
    ASSERT_EQ(space->shape_.size(0), 1);
    ASSERT_TRUE(space->contains(0));
    ASSERT_TRUE(space->contains(4));
    ASSERT_TRUE(!space->contains(5));

    bool ok = true;
    for(int i = 0; i<15; i++)
    {
        ok = ok && space->contains(space->sample());
    }
    ASSERT_TRUE(ok);
}

TEST(Spaces, Discrete2)
{
    lab::spaces::Discrete space(5, 3);
    ASSERT_EQ(space->name_, "Discrete");
    ASSERT_EQ(space->options.n(), 5);
    ASSERT_EQ(space->shape_.size(0), 1);
    ASSERT_TRUE(space->contains(3));
    ASSERT_TRUE(space->contains(7));
    ASSERT_TRUE(!space->contains(8));

    bool ok = true;
    for(int i = 0; i<15; i++)
    {
        ok = ok && space->contains(space->sample());
    }
    ASSERT_TRUE(ok);
}

TEST(Spaces, Discrete3)
{
    lab::spaces::Discrete space1(5, 0);
    lab::spaces::Discrete space2 = lab::spaces::make_discrete_space(6, 3);
    lab::spaces::Discrete space3 = space2;

    ASSERT_EQ( space1->name_, "Discrete" );
    ASSERT_EQ( space1->options.n(), 5 );
    ASSERT_EQ( space1->shape_.size(0), 1 );
    ASSERT_TRUE( space1->contains(0) );
    ASSERT_TRUE( space1->contains(3) );
    ASSERT_TRUE( !space1->contains(5) );

    ASSERT_EQ( space2->name_, "Discrete" );
    ASSERT_EQ( space2->options.n(), 6 );
    ASSERT_EQ( space2->shape_.size(0), 1 );
    ASSERT_TRUE( space2->contains(3) );
    ASSERT_TRUE( space2->contains(6) );
    ASSERT_TRUE( !space2->contains(9) );

    ASSERT_EQ( space3->name_, "Discrete" );
    ASSERT_EQ( space3->options.n(), 6 );
    ASSERT_EQ( space3->shape_.size(0), 1 );
    ASSERT_TRUE( space3->contains(3) );
    ASSERT_TRUE( space3->contains(6) );
    ASSERT_TRUE( !space3->contains(9) );

    bool ok = true;
    for(int i = 0; i<15; i++)
    {
        ok = ok && space2->contains(space2->sample());
    }
    ASSERT_TRUE(ok);
}

