#pragma once

#include <string>
#include <cstdint>

#include "io/image_io.h"
#include "model/model_registry.h"
#include "resolution/resolution_plan.h"

namespace seedvr2
{

// Run the product single-image path for one resolved model variant.
// The output is RGB8 in the requested target dimensions.
bool run_image_inference(const ModelGraphSet& graphs,
                         const RgbImage& input,
                         const ResolutionPlan& plan,
                         int gpu_id,
                         RgbImage& output,
                         std::string& error,
                         std::uint32_t memory_budget_mib = 0);

} // namespace seedvr2
