#pragma once

#include <memory>
#include <string>

#include "gpu.h"
#include "mat.h"
#include "resolution/resolution_plan.h"

namespace ncnn
{
class PipelineCache;
}

namespace seedvr2
{

class PerformanceProfile;

class DitStackSession final
{
public:
    DitStackSession();
    ~DitStackSession();
    DitStackSession(DitStackSession&&) noexcept;
    DitStackSession& operator=(DitStackSession&&) noexcept;
    DitStackSession(const DitStackSession&) = delete;
    DitStackSession& operator=(const DitStackSession&) = delete;

    // Release loaded graph and pipeline resources before allocator reclamation.
    void clear();

    static bool open(const std::string& stack_dir,
                     const ResolutionPlan& plan,
                     ncnn::VulkanDevice* vkdev,
                     ncnn::VkAllocator* blob_allocator,
                     ncnn::VkAllocator* staging_allocator,
                     DitStackSession& session,
                     const PerformanceProfile* profile = nullptr,
                     ncnn::PipelineCache* pipeline_cache = nullptr);

    bool run(const ncnn::VkMat& input_patches,
             const ncnn::Mat& text,
             float timestep_value,
             const ResolutionPlan& plan,
             ncnn::VkMat& output_matrix_gpu) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool make_dit_input_patches_gpu(const ncnn::VkMat& noise,
                                const ncnn::VkMat& condition,
                                const ResolutionPlan& plan,
                                ncnn::VulkanDevice* vkdev,
                                ncnn::VkAllocator* blob_allocator,
                                ncnn::VkAllocator* staging_allocator,
                                ncnn::VkMat& patches,
                                ncnn::PipelineCache* pipeline_cache = nullptr);

bool patch_latent_for_dit_output_gpu(const ncnn::VkMat& latent,
                                     const ResolutionPlan& plan,
                                     ncnn::VulkanDevice* vkdev,
                                     ncnn::VkAllocator* blob_allocator,
                                     ncnn::VkAllocator* staging_allocator,
                                     ncnn::VkMat& patches,
                                     ncnn::PipelineCache* pipeline_cache = nullptr);

// Convert between the 16-channel latent layout and a (T*H*W)x64 DiT output patch matrix.
bool unpatch_dit_output_gpu(const ncnn::VkMat& patches,
                            const ResolutionPlan& plan,
                            ncnn::VulkanDevice* vkdev,
                            ncnn::VkAllocator* blob_allocator,
                            ncnn::VkAllocator* staging_allocator,
                            ncnn::VkMat& latent,
                            ncnn::PipelineCache* pipeline_cache = nullptr);

bool run_dit_stack_gpu(const ncnn::VkMat& input_patches,
                       const ncnn::Mat& text,
                       float timestep_value,
                       const std::string& stack_dir,
                       const ResolutionPlan& plan,
                       ncnn::VulkanDevice* vkdev,
                       ncnn::VkAllocator* blob_allocator,
                       ncnn::VkAllocator* staging_allocator,
                       ncnn::VkMat& output_matrix_gpu);

// Latent-layout bridge for callers that still hold a CPU ncnn::Mat. Product
// callers should assemble input patches on GPU with make_dit_input_patches_gpu.
bool run_dit_stack_gpu(const ncnn::Mat& latent_input,
                       const ncnn::Mat& text,
                       float timestep_value,
                       const std::string& stack_dir,
                       const ResolutionPlan& plan,
                       ncnn::VulkanDevice* vkdev,
                       ncnn::VkAllocator* blob_allocator,
                       ncnn::VkAllocator* staging_allocator,
                       ncnn::VkMat& output_matrix_gpu);

} // namespace seedvr2
