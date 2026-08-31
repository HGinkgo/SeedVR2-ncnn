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

std::string format_profile_total_line(double elapsed_ms, std::uint64_t peak_rss_mib)
{
    std::ostringstream line;
    line << "profile name=total";
    append_ms(line, elapsed_ms);
    line << " peak-rss-mib=" << peak_rss_mib;
    return line.str();
}

std::uint64_t PerformanceProfile::peak_rss_mib() const
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
        if (key == "VmHWM:")
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

} // namespace seedvr2
