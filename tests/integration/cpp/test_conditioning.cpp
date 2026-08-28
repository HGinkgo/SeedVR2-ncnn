#include <cmath>
#include <cstdio>

#include "conditioning/conditioning.h"

int main(int argc, char** argv)
{
    if (argc != 2 && argc != 3)
    {
        std::fprintf(stderr, "usage: seedvr2-conditioning-test <condition-f32>\n");
        return 2;
    }

    const int expected_tokens = argc == 3 ? std::atoi(argv[2]) : seedvr2::kConditioningTokens;
    if (!seedvr2::is_supported_conditioning_tokens(expected_tokens))
    {
        std::fprintf(stderr, "conditioning tokens must be 58 or 64\n");
        return 2;
    }
    ncnn::Mat condition;
    if (!seedvr2::load_conditioning_f32(argv[1], expected_tokens, condition))
    {
        std::fprintf(stderr, "conditioning load failed\n");
        return 1;
    }
    if (condition.dims != 2 || condition.w != seedvr2::kConditioningWidth ||
        condition.h != expected_tokens || condition.elemsize != 4u)
    {
        std::fprintf(stderr, "unexpected condition shape\n");
        return 1;
    }
    const float* data = static_cast<const float*>(condition.data);
    for (size_t index = 0; index < condition.total(); index++)
        if (!std::isfinite(data[index]))
        {
            std::fprintf(stderr, "non-finite condition value\n");
            return 1;
        }

    std::puts("seedvr2-conditioning: ok");
    return 0;
}
