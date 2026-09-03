#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace seedvr2
{

// One stable, machine-parsable profile line, e.g.:
//   profile name=vae-encode ms=123.4
//
// The keyword is deliberately `name=` and not `stage=`: the product path already
// prints `stage=<name>` progress lines that CLI contracts count, and profile
// output must never be mistaken for them.
std::string format_profile_line(const char* name, double elapsed_ms);

// Profile line with one extra count field, e.g.:
//   profile name=vae-encode frame=7 ms=45.25
//   profile name=video-batch frames=2 ms=9876.5
std::string format_profile_line(const char* name,
                                const char* count_label,
                                std::size_t count_value,
                                double elapsed_ms);

// Closing profile line carrying the peak host memory, e.g.:
//   profile name=total ms=13579.0 peak-rss-mib=2913
std::string format_profile_total_line(double elapsed_ms, std::uint64_t peak_rss_mib);

// Aggregate DiT graph-load lines, e.g. `profile name=dit-param-load ms=123.4`.
std::string format_profile_dit_load_line(const char* component, double elapsed_ms);

// Aggregate ncnn model-load stages for the DiT stack, e.g.
// `profile name=dit-ncnn-upload-submit ms=123.4`.
std::string format_profile_dit_stage_line(const char* stage, double elapsed_ms);

// Opt-in stage timing for the Vulkan product path.
//
// Profiling is off unless the caller constructs the profile as enabled. When it
// is off, every entry point is a no-op: no measurement is printed and no
// existing behaviour changes. Nothing here alters allocation policy; the peak
// host memory reading is a passive query of the platform.
class PerformanceProfile final
{
public:
    using Clock = std::chrono::steady_clock;

    PerformanceProfile() = default;
    explicit PerformanceProfile(bool enabled) : enabled_(enabled) {}

    bool enabled() const { return enabled_; }

    double elapsed_ms(Clock::time_point start) const
    {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    void report(const char* name, double elapsed_ms) const
    {
        if (enabled_)
            std::fprintf(stderr, "%s\n", format_profile_line(name, elapsed_ms).c_str());
    }

    void report_frame(const char* name, std::size_t frame_index, double elapsed_ms) const
    {
        if (enabled_)
        {
            std::fprintf(stderr, "%s\n",
                         format_profile_line(name, "frame", frame_index, elapsed_ms).c_str());
        }
    }

    void report_batch(const char* name, std::size_t frame_count, double elapsed_ms) const
    {
        if (enabled_)
        {
            std::fprintf(stderr, "%s\n",
                         format_profile_line(name, "frames", frame_count, elapsed_ms).c_str());
        }
    }

    void report_total(double elapsed_ms) const
    {
        if (enabled_)
        {
            std::fprintf(stderr, "%s\n",
                         format_profile_total_line(elapsed_ms, peak_rss_mib()).c_str());
        }
    }

    // Peak resident host memory in MiB, or 0 when the platform cannot report it.
    std::uint64_t peak_rss_mib() const;

private:
    bool enabled_ = false;
};

// RAII helper that reports the elapsed time of a scope on destruction.
class ProfileScope final
{
public:
    ProfileScope(const PerformanceProfile& profile, const char* name)
        : profile_(profile), name_(name), start_(PerformanceProfile::Clock::now())
    {
    }

    ProfileScope(const PerformanceProfile& profile, const char* name, std::size_t frame_index)
        : profile_(profile), name_(name), start_(PerformanceProfile::Clock::now())
    {
        frame_index_ = frame_index;
        has_frame_ = true;
    }

    ~ProfileScope()
    {
        const double elapsed_ms = profile_.elapsed_ms(start_);
        if (has_frame_)
            profile_.report_frame(name_, frame_index_, elapsed_ms);
        else
            profile_.report(name_, elapsed_ms);
    }

    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;

private:
    const PerformanceProfile& profile_;
    const char* name_;
    PerformanceProfile::Clock::time_point start_;
    std::size_t frame_index_ = 0;
    bool has_frame_ = false;
};

} // namespace seedvr2
