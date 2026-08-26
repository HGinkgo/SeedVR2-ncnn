#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "sampler/sampler.h"

namespace
{

bool near(float actual, float expected, float tolerance = 1e-4f)
{
    return std::fabs(actual - expected) <= tolerance;
}

} // namespace

int main()
{
    const std::vector<float> positive{2.f, 4.f};
    const std::vector<float> negative{1.f, 1.f};
    std::vector<float> guided;
    if (!seedvr2::classifier_free_guidance(positive, negative, 7.5f, 0.f, guided))
        return 1;
    if (guided.size() != 2 || !near(guided[0], 8.5f) || !near(guided[1], 23.5f))
        return 1;

    if (!seedvr2::classifier_free_guidance(positive, negative, 7.5f, 0.f, guided))
        return 1;
    if (!near(guided[0], negative[0] + 7.5f * (positive[0] - negative[0])) ||
        !near(guided[1], negative[1] + 7.5f * (positive[1] - negative[1])))
        return 1;

    if (!seedvr2::classifier_free_guidance({1.f, 3.f}, {0.f, 0.f}, 2.f, 1.f, guided))
        return 1;
    if (!near(guided[0], 1.f) || !near(guided[1], 3.f))
        return 1;

    const std::vector<float> timesteps = seedvr2::uniform_trailing_timesteps();
    if (timesteps.size() != 50 || !near(timesteps.front(), 1000.f) || !near(timesteps.back(), 20.f))
        return 1;
    for (size_t index = 1; index < timesteps.size(); index++)
        if (!(timesteps[index - 1] > timesteps[index]))
            return 1;

    std::vector<float> next;
    if (!seedvr2::euler_v_lerp_step({10.f, 20.f}, {2.f, -4.f}, 1000.f, 500.f, 1000.f, next))
        return 1;
    if (next.size() != 2 || !near(next[0], 9.f) || !near(next[1], 22.f))
        return 1;

    if (seedvr2::classifier_free_guidance({1.f}, {}, 1.f, 0.f, guided))
        return 1;
    if (seedvr2::classifier_free_guidance(positive, negative, std::numeric_limits<float>::quiet_NaN(), 0.f, guided))
        return 1;
    if (!seedvr2::uniform_trailing_timesteps(1000.f, 0, 1.f).empty())
        return 1;
    if (seedvr2::euler_v_lerp_step({1.f}, {1.f}, 10.f, 20.f, 1000.f, next))
        return 1;

    std::puts("seedvr2-sampler: ok");
    return 0;
}
