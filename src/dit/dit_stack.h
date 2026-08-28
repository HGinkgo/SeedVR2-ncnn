#pragma once

#include <string>

#include "gpu.h"
#include "mat.h"
#include "resolution/resolution_plan.h"

namespace seedvr2
{

bool make_dit_input_patches_gpu(const ncnn::VkMat& noise,
                                const ncnn::VkMat& condition,
                                const ResolutionPlan& plan,
                                ncnn::VulkanDevice* vkdev,
                                ncnn::VkAllocator* blob_allocator,
                                ncnn::VkAllocator* staging_allocator,
                                ncnn::VkMat& patches);

// Build the official SeedVR2 input layout:
// (noise[16], VAE condition[16], mask[1]) -> (T*H*W)x132 patch matrix.
bool make_dit_input_patches_gpu(const ncnn::VkMat& noise,
                                const ncnn::VkMat& condition,
                                ncnn::VulkanDevice* vkdev,
                                ncnn::VkAllocator* blob_allocator,
                                ncnn::VkAllocator* staging_allocator,
                                ncnn::VkMat& patches);

bool patch_latent_for_dit_output_gpu(const ncnn::VkMat& latent,
                                     const ResolutionPlan& plan,
                                     ncnn::VulkanDevice* vkdev,
                                     ncnn::VkAllocator* blob_allocator,
                                     ncnn::VkAllocator* staging_allocator,
                                     ncnn::VkMat& patches);

// Convert between the 16-channel latent layout and a (T*H*W)x64 DiT output patch matrix.
bool unpatch_dit_output_gpu(const ncnn::VkMat& patches,
                            const ResolutionPlan& plan,
                            ncnn::VulkanDevice* vkdev,
                            ncnn::VkAllocator* blob_allocator,
                            ncnn::VkAllocator* staging_allocator,
                            ncnn::VkMat& latent);

bool patch_latent_for_dit_output_gpu(const ncnn::VkMat& latent,
                                     ncnn::VulkanDevice* vkdev,
                                     ncnn::VkAllocator* blob_allocator,
                                     ncnn::VkAllocator* staging_allocator,
                                     ncnn::VkMat& patches);

bool unpatch_dit_output_gpu(const ncnn::VkMat& patches,
                            ncnn::VulkanDevice* vkdev,
                            ncnn::VkAllocator* blob_allocator,
                            ncnn::VkAllocator* staging_allocator,
                            ncnn::VkMat& latent);

bool run_dit_stack_gpu(const ncnn::VkMat& input_patches,
                       const ncnn::Mat& text,
                       float timestep_value,
                       const std::string& stack_dir,
                       const ResolutionPlan& plan,
                       ncnn::VulkanDevice* vkdev,
                       ncnn::VkAllocator* blob_allocator,
                       ncnn::VkAllocator* staging_allocator,
                       ncnn::VkMat& output_matrix_gpu);

bool run_dit_stack_gpu(const ncnn::VkMat& input_patches,
                       const ncnn::Mat& text,
                       float timestep_value,
                       const std::string& stack_dir,
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

bool run_dit_stack_gpu(const ncnn::Mat& latent_input,
                       const ncnn::Mat& text,
                       float timestep_value,
                       const std::string& stack_dir,
                       ncnn::VulkanDevice* vkdev,
                       ncnn::VkAllocator* blob_allocator,
                       ncnn::VkAllocator* staging_allocator,
                       ncnn::VkMat& output_matrix_gpu);

} // namespace seedvr2
