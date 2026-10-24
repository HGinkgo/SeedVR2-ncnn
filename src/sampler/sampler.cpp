#include "sampler/sampler.h"

#include <cmath>
#include <random>

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

bool fill_gaussian_noise(float* output, std::size_t count, std::uint32_t seed,
                         std::uint64_t stream)
{
    if (!output || count == 0)
        return false;

    std::seed_seq sequence{seed, static_cast<std::uint32_t>(stream),
                           static_cast<std::uint32_t>(stream >> 32)};
    std::mt19937 generator(sequence);
    constexpr double inverse_range = 1.0 / 4294967296.0;
    constexpr double two_pi = 6.283185307179586476925286766559;
    for (std::size_t index = 0; index < count; index += 2)
    {
        // Midpoints keep both uniforms strictly inside (0,1), including the
        // generator's endpoints. Do the transform in double before FP32 cast.
        const double radial_uniform = (static_cast<double>(generator()) + 0.5) * inverse_range;
        const double angle_uniform = (static_cast<double>(generator()) + 0.5) * inverse_range;
        const double radius = std::sqrt(-2.0 * std::log(radial_uniform));
        const double angle = two_pi * angle_uniform;
        output[index] = static_cast<float>(radius * std::cos(angle));
        if (index + 1 < count)
            output[index + 1] = static_cast<float>(radius * std::sin(angle));
    }
    return true;
}

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

bool transform_timestep(float timestep,
                        float schedule_t,
                        int latent_frames,
                        int latent_height,
                        int latent_width,
                        float& transformed)
{
    if (!std::isfinite(timestep) || !std::isfinite(schedule_t) || schedule_t <= 0.f ||
        timestep < 0.f || timestep > schedule_t || latent_frames <= 0 || latent_height <= 0 ||
        latent_width <= 0)
        return false;

    // The fixed 256x256 image contract is exactly the reference image anchor
    // (shift=1). Preserve it bit-for-bit for the existing production path.
    if (latent_frames == 1 && latent_height == 32 && latent_width == 32)
    {
        transformed = timestep;
        return true;
    }

    constexpr float temporal_downsample = 4.f;
    constexpr float spatial_downsample = 8.f;
    const float frames = (static_cast<float>(latent_frames) - 1.f) * temporal_downsample + 1.f;
    const float height = static_cast<float>(latent_height) * spatial_downsample;
    const float width = static_cast<float>(latent_width) * spatial_downsample;
    const float image_m = (3.2f - 1.f) / (1024.f * 1024.f - 256.f * 256.f);
    const float image_b = 1.f - image_m * 256.f * 256.f;
    const float video_m = (5.f - 1.f) /
                          (1280.f * 720.f * 145.f - 256.f * 256.f * 37.f);
    const float video_b = 1.f - video_m * 256.f * 256.f * 37.f;
    const float image_shift = image_m * height * width + image_b;
    const float video_shift = video_m * height * width * frames + video_b;
    const float shift = frames > 1.f ? video_shift : image_shift;
    if (!std::isfinite(shift) || shift <= 0.f)
        return false;

    const float normalized = timestep / schedule_t;
    const float denominator = 1.f + (shift - 1.f) * normalized;
    if (!std::isfinite(denominator) || denominator <= 0.f)
        return false;
    transformed = schedule_t * shift * normalized / denominator;
    return std::isfinite(transformed) && transformed >= 0.f && transformed <= schedule_t;
}

bool euler_v_lerp_endpoint(const std::vector<float>& sample,
                           const std::vector<float>& model_output,
                           float timestep,
                           float schedule_t,
                           std::vector<float>& endpoint)
{
    if (sample.empty() || sample.size() != model_output.size() || !all_finite(sample) ||
        !all_finite(model_output) || !std::isfinite(timestep) || !std::isfinite(schedule_t) ||
        schedule_t <= 0.f || timestep != schedule_t)
        return false;

    endpoint.resize(sample.size());
    for (size_t index = 0; index < sample.size(); index++)
        endpoint[index] = sample[index] - model_output[index];
    return all_finite(endpoint);
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
