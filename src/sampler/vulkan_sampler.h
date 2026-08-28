#pragma once

#include "gpu.h"

namespace seedvr2
{

// DiT predictions must be pack1. The sample may be pack1 or pack4; the result is pack1.
bool apply_cfg_euler_vulkan(const ncnn::VkMat& positive_output,
                            const ncnn::VkMat& negative_output,
                            const ncnn::VkMat& sample,
                            float cfg_scale,
                            float normalized_delta,
                            ncnn::VulkanDevice* vkdev,
                            ncnn::VkAllocator* blob_allocator,
                            ncnn::VkAllocator* staging_allocator,
                            ncnn::VkMat& updated_sample);

// Official one-step path: CFG scale one uses only the positive prediction and
// resolves v_lerp at t=T as x0 = sample - prediction.
bool apply_cfg_v_lerp_endpoint_vulkan(const ncnn::VkMat& positive_output,
                                       const ncnn::VkMat& sample,
                                       ncnn::VulkanDevice* vkdev,
                                       ncnn::VkAllocator* blob_allocator,
                                       ncnn::VkAllocator* staging_allocator,
                                       ncnn::VkMat& endpoint_sample);

} // namespace seedvr2
