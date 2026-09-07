#include "performance_profile.h"

#include <cstdio>
#include <fstream>
#include <sstream>

namespace seedvr2
{

namespace
{

void append_ms(std::ostringstream& line, double elapsed_ms)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.1f", elapsed_ms);
    line << " ms=" << buffer;
}

} // namespace

std::string format_profile_line(const char* name, double elapsed_ms)
{
    std::ostringstream line;
    line << "profile name=" << name;
    append_ms(line, elapsed_ms);
    return line.str();
}

std::string format_profile_line(const char* name,
                                const char* count_label,
                                std::size_t count_value,
                                double elapsed_ms)
{
    std::ostringstream line;
    line << "profile name=" << name << ' ' << count_label << '=' << count_value;
    append_ms(line, elapsed_ms);
    return line.str();
}

std::string format_profile_mode_line(const char* name, const char* mode)
{
    std::ostringstream line;
    line << "profile name=" << name << " mode=" << mode;
    return line.str();
}

std::string format_profile_residency_line(const char* phase,
                                          std::uint64_t rss_mib,
                                          std::uint64_t peak_rss_mib,
                                          std::uint32_t heap_budget_mib,
                                          std::uint64_t max_allocation_mib)
{
    std::ostringstream line;
    line << "profile name=residency phase=" << phase << " rss-mib=" << rss_mib
         << " peak-rss-mib=" << peak_rss_mib << " heap-budget-mib=" << heap_budget_mib
         << " max-allocation-mib=" << max_allocation_mib;
    return line.str();
}

std::string format_profile_session_open_line(const char* mode, double elapsed_ms)
{
    std::ostringstream line;
    line << "profile name=session-open mode=" << mode;
    append_ms(line, elapsed_ms);
    return line.str();
}

std::string format_profile_session_run_line(const char* mode,
                                            std::size_t run_index,
                                            double elapsed_ms)
{
    std::ostringstream line;
    line << "profile name=session-run mode=" << mode << " index=" << run_index;
    append_ms(line, elapsed_ms);
    return line.str();
}

std::string format_profile_pipeline_cache_line(const char* phase, std::size_t entries)
{
    std::ostringstream line;
    line << "profile name=pipeline-cache phase=" << phase << " entries=" << entries;
    return line.str();
}

std::string format_profile_total_line(double elapsed_ms, std::uint64_t peak_rss_mib)
{
    std::ostringstream line;
    line << "profile name=total";
    append_ms(line, elapsed_ms);
    line << " peak-rss-mib=" << peak_rss_mib;
    return line.str();
}

std::string format_profile_dit_load_line(const char* component, double elapsed_ms)
{
    std::ostringstream line;
    line << "profile name=dit-" << component << "-load";
    append_ms(line, elapsed_ms);
    return line.str();
}

std::string format_profile_dit_stage_line(const char* stage, double elapsed_ms)
{
    std::ostringstream line;
    line << "profile name=dit-" << stage;
    append_ms(line, elapsed_ms);
    return line.str();
}

namespace
{

std::uint64_t read_proc_status_mib(const char* field)
{
#if defined(__linux__)
    // Read the high-water mark the kernel already tracks for us. This is a
    // passive query and never changes how memory is allocated.
    std::ifstream status("/proc/self/status");
    if (!status)
        return 0;

    std::string key;
    while (status >> key)
    {
        if (key == field)
        {
            std::uint64_t kib = 0;
            if (status >> kib)
                return kib / 1024;
            return 0;
        }
        std::string rest;
        std::getline(status, rest);
    }
#endif
    return 0;
}

} // namespace

std::uint64_t PerformanceProfile::peak_rss_mib() const
{
    return read_proc_status_mib("VmHWM:");
}

std::uint64_t PerformanceProfile::current_rss_mib() const
{
    return read_proc_status_mib("VmRSS:");
}

} // namespace seedvr2
