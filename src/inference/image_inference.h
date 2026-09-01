#pragma once

#include <cstddef>
#include <string>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "io/image_io.h"
#include "inference/performance_profile.h"
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

    // `profile` is optional and defaults to disabled: it only enables the extra
    // `--profile` stderr lines and never changes inference behaviour. The caller
    // must keep the profile alive for as long as the session is in use.
    static bool open(const ModelGraphSet& graphs,
                     const ResolutionPlan& plan,
                     int gpu_id,
                     ImageInferenceSession& session,
                     std::string& error,
                     std::uint32_t memory_budget_mib = 0,
                     const PerformanceProfile* profile = nullptr);

    bool run_frame(const RgbImage& input, RgbImage& output, std::string& error) const;

    // `frame_offset` is the index of the first frame of this batch within the
    // whole clip, used only to label profile lines.
    bool run_batch(const std::vector<RgbImage>& inputs,
                   std::vector<RgbImage>& outputs,
                   std::string& error,
                   std::size_t frame_offset = 0) const;

    using VideoFrameReader = std::function<bool(RgbImage&, std::string&)>;
    using VideoFrameWriter = std::function<bool(const RgbImage&, std::string&)>;

    // Process a sequential video stream while loading each model group only
    // once. A reader returning false with an empty error denotes end-of-file.
    bool run_video(const VideoFrameReader& reader,
                   const VideoFrameWriter& writer,
                   std::size_t& frame_count,
                   std::string& error,
                   std::size_t frame_offset = 0) const;

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
                         std::uint32_t memory_budget_mib = 0,
                         const PerformanceProfile* profile = nullptr);

} // namespace seedvr2
