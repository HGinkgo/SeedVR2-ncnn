#pragma once

#include <cstddef>
#include <string>
#include <cstdint>
#include <memory>
#include <vector>

#include "io/image_io.h"
#include "model/model_registry.h"
#include "resolution/resolution_plan.h"

namespace seedvr2
{

class ImageInferenceSession final
{
public:
    static constexpr std::size_t kMaxBatchFrames = 2;

    ImageInferenceSession();
    ~ImageInferenceSession();
    ImageInferenceSession(ImageInferenceSession&&) noexcept;
    ImageInferenceSession& operator=(ImageInferenceSession&&) noexcept;
    ImageInferenceSession(const ImageInferenceSession&) = delete;
    ImageInferenceSession& operator=(const ImageInferenceSession&) = delete;

    static bool open(const ModelGraphSet& graphs,
                     const ResolutionPlan& plan,
                     int gpu_id,
                     ImageInferenceSession& session,
                     std::string& error,
                     std::uint32_t memory_budget_mib = 0);

    bool run_frame(const RgbImage& input, RgbImage& output, std::string& error) const;
    bool run_batch(const std::vector<RgbImage>& inputs,
                   std::vector<RgbImage>& outputs,
                   std::string& error) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Run the product single-image path for one resolved model package.
// The output is RGB8 in the requested target dimensions.
bool run_image_inference(const ModelGraphSet& graphs,
                         const RgbImage& input,
                         const ResolutionPlan& plan,
                         int gpu_id,
                         RgbImage& output,
                         std::string& error,
                         std::uint32_t memory_budget_mib = 0);

} // namespace seedvr2
