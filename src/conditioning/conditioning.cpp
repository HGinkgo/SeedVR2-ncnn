#include "conditioning/conditioning.h"

#include <cstdint>
#include <fstream>
#include <limits>

namespace seedvr2
{

bool load_conditioning_f32(const char* path, ncnn::Mat& condition)
{
    return load_conditioning_f32(path, kConditioningTokens, condition);
}

bool load_conditioning_f32(const char* path, int tokens, ncnn::Mat& condition)
{
    if (!path || !is_supported_conditioning_tokens(tokens))
        return false;

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        return false;
    const std::streamoff expected_bytes = static_cast<std::streamoff>(tokens) *
                                          static_cast<std::streamoff>(kConditioningWidth) *
                                          static_cast<std::streamoff>(sizeof(float));
    if (input.tellg() != expected_bytes)
        return false;
    input.seekg(0, std::ios::beg);

    ncnn::Mat loaded(kConditioningWidth, tokens);
    if (loaded.empty())
        return false;
    input.read(static_cast<char*>(loaded.data), expected_bytes);
    if (!input || input.gcount() != expected_bytes)
        return false;

    condition = loaded;
    return true;
}

} // namespace seedvr2
