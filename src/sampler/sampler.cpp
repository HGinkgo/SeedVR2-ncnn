#include "sampler/sampler.h"

#include <cmath>

namespace seedvr2
{
namespace
{

bool all_finite(const std::vector<float>& values)
{
    for (const float value : values)
        if (!std::isfinite(value))
            return false;
    return true;
}

float sample_stddev(const std::vector<float>& values)
{
    float mean = 0.f;
    for (const float value : values)
        mean += value;
    mean /= static_cast<float>(values.size());

    float squared_error = 0.f;
    for (const float value : values)
    {
        const float delta = value - mean;
        squared_error += delta * delta;
    }
    return std::sqrt(squared_error / static_cast<float>(values.size() - 1));
}

} // namespace

bool classifier_free_guidance(const std::vector<float>& positive,
                              const std::vector<float>& negative,
                              float scale,
                              float rescale,
                              std::vector<float>& guided)
{
    if (positive.empty() || positive.size() != negative.size() || !std::isfinite(scale) ||
        !std::isfinite(rescale) || rescale < 0.f || rescale > 1.f || !all_finite(positive) ||
        !all_finite(negative))
        return false;

    guided.resize(positive.size());
    for (size_t index = 0; index < guided.size(); index++)
        guided[index] = negative[index] + scale * (positive[index] - negative[index]);

    if (rescale == 0.f)
        return all_finite(guided);
    if (guided.size() < 2)
        return false;

    const float positive_std = sample_stddev(positive);
    const float guided_std = sample_stddev(guided);
    if (!std::isfinite(positive_std) || !std::isfinite(guided_std) || guided_std == 0.f)
        return false;

    const float factor = rescale * positive_std / guided_std + (1.f - rescale);
    for (float& value : guided)
        value *= factor;
    return all_finite(guided);
}

std::vector<float> uniform_trailing_timesteps(float schedule_t, int steps, float shift)
{
    if (steps <= 0 || !std::isfinite(schedule_t) || schedule_t <= 0.f || !std::isfinite(shift) ||
        shift <= 0.f)
        return {};

    std::vector<float> timesteps;
    timesteps.reserve(static_cast<size_t>(steps));
    for (int index = 0; index < steps; index++)
    {
        const float unit = 1.f - static_cast<float>(index) / static_cast<float>(steps);
        const float shifted = shift * unit / (1.f + (shift - 1.f) * unit);
        timesteps.push_back(schedule_t * shifted);
    }
    return timesteps;
}

bool euler_v_lerp_step(const std::vector<float>& sample,
                       const std::vector<float>& model_output,
                       float timestep,
                       float next_timestep,
                       float schedule_t,
                       std::vector<float>& next_sample)
{
    if (sample.empty() || sample.size() != model_output.size() || !all_finite(sample) ||
        !all_finite(model_output) || !std::isfinite(timestep) || !std::isfinite(next_timestep) ||
        !std::isfinite(schedule_t) || schedule_t <= 0.f || timestep < 0.f ||
        timestep > schedule_t || next_timestep < 0.f || next_timestep > schedule_t ||
        next_timestep > timestep)
        return false;

    const float delta = (next_timestep - timestep) / schedule_t;
    next_sample.resize(sample.size());
    for (size_t index = 0; index < sample.size(); index++)
        next_sample[index] = sample[index] + delta * model_output[index];
    return all_finite(next_sample);
}

} // namespace seedvr2
