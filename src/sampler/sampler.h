#pragma once

#include <vector>

namespace seedvr2
{

bool classifier_free_guidance(const std::vector<float>& positive,
                              const std::vector<float>& negative,
                              float scale,
                              float rescale,
                              std::vector<float>& guided);

std::vector<float> uniform_trailing_timesteps(float schedule_t = 1000.f,
                                              int steps = 1,
                                              float shift = 1.f);

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
