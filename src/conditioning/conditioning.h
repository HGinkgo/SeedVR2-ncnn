#pragma once

#include "mat.h"

namespace seedvr2
{

constexpr int kConditioningTokens = 58;
constexpr int kNegativeConditioningTokens = 64;
constexpr int kConditioningWidth = 5120;

constexpr bool is_supported_conditioning_tokens(int tokens)
{
    return tokens == kConditioningTokens || tokens == kNegativeConditioningTokens;
}

bool load_conditioning_f32(const char* path, ncnn::Mat& condition);
bool load_conditioning_f32(const char* path, int tokens, ncnn::Mat& condition);

} // namespace seedvr2
