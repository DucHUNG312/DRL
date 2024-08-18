#pragma once

#include "lab/core.h"

namespace lab
{
namespace utils
{

class DataFrame
{
public:
    DataFrame() = default;
    ~DataFrame() = default;

    void load_df_columns(const std::vector<const char*>& attrs)
    {

        df.load_index(std::move(ULDataFrame::gen_sequence_index(0, attrs.size() - 1 , 1)));
        for(const auto& attr : attrs)
            df.load_column<double>(attr, std::move(std::vector<double>{}));
    }

    hmdf::HeteroVector<0> get_row(uint64_t idx, const std::vector<const char*>& attrs)
    {
        LAB_CHECK_LT(idx, df.shape().second);
        return df.get_row<double>(idx, attrs);
    }

    hmdf::HeteroVector<0> get_last_row(const std::vector<const char*>& attrs)
    {
        return get_row(df.shape().second-1, attrs);
    }

    void modify_row(uint64_t idx, const std::vector<const char*>& attrs, const std::vector<double>& values)
    {
        LAB_CHECK_LT(idx, df.shape().second);
        for(int64_t i = 0; i < attrs.size(); i++)
        {
            auto colvals = df.get_column<double>(attrs[i]);
            colvals[idx] = values[i];
        }
    }
public:
    ULDataFrame df;
};

}
}