#pragma once

#include "allocator.h"

namespace seedvr2
{

// CPU-only layers force synchronous GPU/CPU transfers. Keeping completed
// transfer buffers cached across a large VAE decode needlessly retains memory.
class TransientVkStagingAllocator final : public ncnn::VkStagingAllocator
{
public:
    explicit TransientVkStagingAllocator(const ncnn::VulkanDevice* vkdev)
        : ncnn::VkStagingAllocator(vkdev)
    {
    }

    void fastFree(ncnn::VkBufferMemory* ptr) override
    {
        ncnn::VkStagingAllocator::fastFree(ptr);
        ncnn::VkStagingAllocator::clear();
    }
};

} // namespace seedvr2
