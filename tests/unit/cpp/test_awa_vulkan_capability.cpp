#include <cstdio>

#include "awa/awa_layers.h"

int main()
{
    SeedVR2AWAPack pack;
    SeedVR2AWAUnpack unpack;

    if (!pack.support_vulkan || !unpack.support_vulkan)
    {
        std::fprintf(stderr, "AWA layers must enable Vulkan support\n");
        return 1;
    }
    if (pack.support_vulkan_packing || unpack.support_vulkan_packing)
    {
        std::fprintf(stderr, "AWA Vulkan path must initially require pack1 tensors\n");
        return 1;
    }

    std::puts("seedvr2-awa-vulkan-capability: ok");
    return 0;
}
