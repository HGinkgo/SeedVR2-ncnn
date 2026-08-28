#include "memory_diagnostics.h"

#include <sstream>

namespace seedvr2
{

std::string format_vulkan_stage_error(const std::string& stage,
                                      const VulkanMemoryDiagnostics& diagnostics,
                                      const std::string& detail)
{
    std::ostringstream message;
    message << "stage=" << stage << " failed: " << detail << " (gpu=" << diagnostics.gpu_id;
    if (!diagnostics.device_name.empty())
        message << ' ' << diagnostics.device_name;
    message << ", heap-budget-mib=" << diagnostics.heap_budget_mib
            << ", max-allocation-mib=" << diagnostics.max_allocation_mib << ", target="
            << diagnostics.target_width << 'x' << diagnostics.target_height
            << "; possible Vulkan memory exhaustion; reduce --width/--height, choose a smaller model variant, or select another GPU)";
    return message.str();
}

std::string format_vulkan_memory_preflight_error(const VulkanMemoryDiagnostics& diagnostics,
                                                 std::uint32_t requested_budget_mib)
{
    std::ostringstream message;
    message << "Vulkan memory preflight failed: requested minimum " << requested_budget_mib
            << " MiB exceeds device heap budget " << diagnostics.heap_budget_mib << " MiB (gpu="
            << diagnostics.gpu_id;
    if (!diagnostics.device_name.empty())
        message << ' ' << diagnostics.device_name;
    message << ", target=" << diagnostics.target_width << 'x' << diagnostics.target_height
            << "; lower --memory-budget-mib, reduce --width/--height, or select another GPU)";
    return message.str();
}

} // namespace seedvr2
