#pragma once

#include <cstdint>

#include "net.h"

namespace seedvr2
{

inline bool prepare_ncnn_layer_benchmark(ncnn::VkCompute& compute, const ncnn::Net& net)
{
#if NCNN_BENCHMARK
    return compute.create_query_pool(static_cast<uint32_t>(net.layers().size() * 2)) == 0;
#else
    (void)compute;
    (void)net;
    return true;
#endif
}

} // namespace seedvr2
