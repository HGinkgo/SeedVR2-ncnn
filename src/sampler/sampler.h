#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace seedvr2
{

// Standard-normal FP32 values from mt19937/seed_seq and Box-Muller. A stream
// is an absolute frame index, so batching and video slicing do not change it.
// Reproducibility is scoped to the same runtime build, not PyTorch's RNG.
bool fill_gaussian_noise(float* output, std::size_t count, std::uint32_t seed,
                         std::uint64_t stream = 0);

bool classifier_free_guidance(const std::vector<float>& positive,
                              const std::vector<float>& negative,
                              float scale,
                              float rescale,
                              std::vector<float>& guided);

std::vector<float> uniform_trailing_timesteps(float schedule_t = 1000.f,
                                              int steps = 1,
                                              float shift = 1.f);

// Apply SeedVR2's resolution-dependent timestep transform before the DiT.
// The dimensions are the latent shape [frames, height, width].
bool transform_timestep(float timestep,
                        float schedule_t,
                        int latent_frames,
                        int latent_height,
                        int latent_width,
                        float& transformed);

bool euler_v_lerp_endpoint(const std::vector<float>& sample,
                           const std::vector<float>& model_output,
                           float timestep,
                           float schedule_t,
                           std::vector<float>& endpoint);

bool euler_v_lerp_step(const std::vector<float>& sample,
                       const std::vector<float>& model_output,
                       float timestep,
                       float next_timestep,
                       float schedule_t,
                       std::vector<float>& next_sample);

} // namespace seedvr2
