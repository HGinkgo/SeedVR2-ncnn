#pragma once

#include <cstdint>
#include <string>

namespace seedvr2
{

struct VulkanMemoryDiagnostics final
{
    int gpu_id = -1;
    std::string device_name;
    std::uint32_t heap_budget_mib = 0;
    std::uint64_t max_allocation_mib = 0;
    int target_width = 0;
    int target_height = 0;
};

std::string format_vulkan_stage_error(const std::string& stage,
                                      const VulkanMemoryDiagnostics& diagnostics,
                                      const std::string& detail);

std::string format_vulkan_memory_preflight_error(const VulkanMemoryDiagnostics& diagnostics,
                                                 std::uint32_t requested_budget_mib);

} // namespace seedvr2
