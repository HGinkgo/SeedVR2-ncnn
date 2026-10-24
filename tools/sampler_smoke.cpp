// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 SeedVR2-ncnn contributors

#include "sampler/sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{

void require(bool condition, const char* message)
{
    if (!condition)
    {
        std::fprintf(stderr, "sampler-smoke: failure: %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

void require_close(float actual, float expected, float tolerance, const char* message)
{
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance)
    {
        std::fprintf(stderr, "sampler-smoke: failure: %s actual=%.9g expected=%.9g tolerance=%.9g\n",
                     message, actual, expected, tolerance);
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main()
{
    // Check distribution as well as seed/stream determinism. A uniform
    // fixture or reused frame stream must not pass this contract.
    constexpr std::size_t noise_count = 131073;
    std::vector<float> noise(noise_count + 1, 9876.f);
    std::vector<float> repeat(noise_count);
    require(seedvr2::fill_gaussian_noise(noise.data(), noise_count, 666, 7), "Gaussian fill");
    require(noise.back() == 9876.f, "odd-size fill must preserve tail sentinel");
    require(seedvr2::fill_gaussian_noise(repeat.data(), repeat.size(), 666, 7), "Gaussian repeat");
    require(std::equal(repeat.begin(), repeat.end(), noise.begin()), "same seed and stream repeat exactly");
    double mean = 0.0;
    double square_mean = 0.0;
    std::size_t tails = 0;
    for (std::size_t index = 0; index < noise_count; ++index)
    {
        require(std::isfinite(noise[index]), "Gaussian samples must be finite");
        mean += noise[index];
        square_mean += static_cast<double>(noise[index]) * noise[index];
        tails += std::fabs(noise[index]) > 3.f;
    }
    mean /= noise_count;
    const double variance = square_mean / noise_count - mean * mean;
    require(std::fabs(mean) < 0.02 && std::fabs(variance - 1.0) < 0.04,
            "Gaussian mean and variance");
    require(tails > 100 && tails < 700, "Gaussian tails");
    for (const std::uint64_t stream : {8ULL, (1ULL << 32) + 7})
    {
        require(seedvr2::fill_gaussian_noise(repeat.data(), repeat.size(), 666, stream), "frame stream");
        require(!std::equal(repeat.begin(), repeat.end(), noise.begin()), "distinct 64-bit frame streams");
    }
    require(seedvr2::fill_gaussian_noise(repeat.data(), repeat.size(), 667, 7), "different seed");
    require(!std::equal(repeat.begin(), repeat.end(), noise.begin()), "distinct seeds");
    require(!seedvr2::fill_gaussian_noise(nullptr, 3, 0) &&
                !seedvr2::fill_gaussian_noise(noise.data(), 0, 0), "invalid noise buffer");
    std::printf("Gaussian seed=666 stream=7 count=%zu mean=%.9g variance=%.9g tails_gt3=%zu\n",
                 noise_count, mean, variance, tails);

    const std::vector<float> timesteps = seedvr2::uniform_trailing_timesteps(1000.f, 2);
    require(timesteps.size() == 2, "two-step schedule size");
    require_close(timesteps[0], 1000.f, 1.0e-6f, "two-step first timestep");
    require_close(timesteps[1], 500.f, 1.0e-6f, "two-step second timestep");

    float transformed = -1.f;
    require(seedvr2::transform_timestep(1000.f, 1000.f, 1, 32, 32, transformed),
            "fixed 256 transform accepted");
    require_close(transformed, 1000.f, 1.0e-6f, "fixed 256 t=1000 identity");
    require(seedvr2::transform_timestep(500.f, 1000.f, 1, 32, 32, transformed),
            "fixed 256 midpoint accepted");
    require_close(transformed, 500.f, 1.0e-6f, "fixed 256 midpoint identity");

    require(seedvr2::transform_timestep(500.f, 1000.f, 1, 64, 64, transformed),
            "512 image transform accepted");
    require_close(transformed, 590.16394f, 1.0e-3f, "512 image midpoint transform");

    require(seedvr2::transform_timestep(500.f, 1000.f, 37, 90, 160, transformed),
            "37-frame 1280x720 transform accepted");
    require_close(transformed, 833.33331f, 1.0e-3f, "video midpoint transform");

    require(!seedvr2::transform_timestep(-1.f, 1000.f, 1, 32, 32, transformed),
            "negative timestep rejected");
    require(!seedvr2::transform_timestep(1001.f, 1000.f, 1, 32, 32, transformed),
            "timestep above schedule rejected");
    require(!seedvr2::transform_timestep(500.f, 0.f, 1, 32, 32, transformed),
            "non-positive schedule rejected");
    require(!seedvr2::transform_timestep(500.f, 1000.f, 0, 32, 32, transformed),
            "invalid temporal shape rejected");

    std::printf("sampler-smoke: ok\n");
    return EXIT_SUCCESS;
}
