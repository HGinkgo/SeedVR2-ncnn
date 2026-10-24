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

// Native BF16 RNE overrides use ncnn's scalar non-SLM Gemm family so each
// BF16 write can be rounded explicitly before storage.
void configure_dit_vulkan_net_rne(ncnn::Net& net,
                                  ncnn::VulkanDevice* vkdev,
                                  ncnn::VkAllocator* blob_allocator,
                                  ncnn::VkAllocator* staging_allocator,
                                  ncnn::PipelineCache* pipeline_cache);

// PT2 stack graphs are validated with scalar FP32 storage.  Keep this opt-in
// so the existing TorchScript/package path retains its BF16 runtime contract.
void configure_dit_vulkan_net_fp32(ncnn::Net& net,
                                   ncnn::VulkanDevice* vkdev,
                                   ncnn::VkAllocator* blob_allocator,
                                   ncnn::VkAllocator* staging_allocator,
                                   ncnn::PipelineCache* pipeline_cache);

} // namespace seedvr2
