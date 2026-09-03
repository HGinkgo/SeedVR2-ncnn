#include <cstdio>

#include "dit/dit_stack_internal.h"
#include "gpu.h"
#include "net.h"
#include "pipelinecache.h"

int main()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
    {
        std::fprintf(stderr, "no Vulkan device available\n");
        return 1;
    }

    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();
    if (!blob_allocator || !staging_allocator)
    {
        std::fprintf(stderr, "allocator acquisition failed\n");
        return 1;
    }

    ncnn::PipelineCache pipeline_cache(vkdev);
    ncnn::Net net;
    seedvr2::configure_dit_vulkan_net(net, vkdev, blob_allocator, staging_allocator, &pipeline_cache);
    if (net.opt.pipeline_cache != &pipeline_cache)
    {
        std::fprintf(stderr, "stage=dit-pipeline-cache-config failed\n");
        return 1;
    }

    net.clear();
    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    return 0;
}
