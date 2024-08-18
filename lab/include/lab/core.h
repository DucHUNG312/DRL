#pragma once

#include <common/common.h>

#include <torch/torch.h>
#include <DataFrame/DataFrame.h>

// A DataFrame with ulong index type
using ULDataFrame = hmdf::StdDataFrame<unsigned long>;
// A DataFrame with string index type
using StrDataFrame = hmdf::StdDataFrame<std::string>;
// A DataFrame with DateTime index type
using DTDataFrame = hmdf::StdDataFrame<hmdf::DateTime>;

#include "lab/version.h"
