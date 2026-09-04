#pragma once

#include <string>

#include "io/image_io.h"
#include "resolution/resolution_plan.h"

namespace seedvr2
{

// Builds the RGB reference at the same resized/cropped dimensions used by the model input.
bool prepare_color_reference(const RgbImage& input, const ResolutionPlan& plan, RgbImage& reference,
                            std::string& error);

// Replaces the generated low frequencies with those from the RGB reference while retaining generated detail.
bool apply_wavelet_color_fix(const RgbImage& content, const RgbImage& reference, RgbImage& output,
                             std::string& error);

} // namespace seedvr2
