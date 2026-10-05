#pragma once

#include <cstdint>

#include "net.h"
#include "inference/performance_profile.h"

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

inline int submit_and_wait_profiled(ncnn::VkCompute& compute, const PerformanceProfile* profile)
{
    if (!profile || !profile->enabled())
        return compute.submit_and_wait();

    const auto start = PerformanceProfile::Clock::now();
    const int result = compute.submit_and_wait();
    profile->record_runtime_submit(profile->elapsed_ms(start));
    return result;
}

} // namespace seedvr2
