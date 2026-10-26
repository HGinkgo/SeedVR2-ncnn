#pragma once

#include "gpu.h"

namespace ncnn
{
class Net;
class PipelineCache;
} // namespace ncnn

namespace seedvr2
{

// Graph-local FP32 sinusoidal prefix; the embedding MLP retains net precision.
void register_dit_timestep_layers(ncnn::Net& net);

void configure_dit_vulkan_net(ncnn::Net& net,
                              ncnn::VulkanDevice* vkdev,
                              ncnn::VkAllocator* blob_allocator,
                              ncnn::VkAllocator* staging_allocator,
                              ncnn::PipelineCache* pipeline_cache);

// The DiT prefix uses scalar FP32 storage while the remaining stack uses BF16.
void configure_dit_vulkan_net_fp32(ncnn::Net& net,
                                   ncnn::VulkanDevice* vkdev,
                                   ncnn::VkAllocator* blob_allocator,
                                   ncnn::VkAllocator* staging_allocator,
                                   ncnn::PipelineCache* pipeline_cache);

} // namespace seedvr2
