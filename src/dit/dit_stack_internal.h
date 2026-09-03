#pragma once

#include "gpu.h"

namespace ncnn
{
class Net;
class PipelineCache;
} // namespace ncnn

namespace seedvr2
{

void configure_dit_vulkan_net(ncnn::Net& net,
                              ncnn::VulkanDevice* vkdev,
                              ncnn::VkAllocator* blob_allocator,
                              ncnn::VkAllocator* staging_allocator,
                              ncnn::PipelineCache* pipeline_cache);

} // namespace seedvr2
