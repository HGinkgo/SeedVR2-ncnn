#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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

// Profile line carrying a stable non-timing mode, e.g.:
//   profile name=vae-graph mode=static-256|static-shape|dynamic
std::string format_profile_mode_line(const char* name, const char* mode);

// Residency checkpoint, e.g.:
//   profile name=residency phase=dit-released rss-mib=512 peak-rss-mib=2048 heap-budget-mib=23676 max-allocation-mib=4094
std::string format_profile_residency_line(const char* phase,
                                          std::uint64_t rss_mib,
                                          std::uint64_t peak_rss_mib,
                                          std::uint32_t heap_budget_mib,
                                          std::uint64_t max_allocation_mib);

// Session lifecycle profile lines, e.g.:
//   profile name=session-open mode=cold ms=123.4
//   profile name=session-run mode=warm index=1 ms=456.7
std::string format_profile_session_open_line(const char* mode, double elapsed_ms);
std::string format_profile_session_run_line(const char* mode,
                                            std::size_t run_index,
                                            double elapsed_ms);
std::string format_profile_pipeline_cache_line(const char* phase, std::size_t entries);
std::string format_profile_model_cache_line(const char* phase, const char* mode);

// Closing profile line carrying the peak host memory, e.g.:
//   profile name=total ms=13579.0 peak-rss-mib=2913
std::string format_profile_total_line(double elapsed_ms, std::uint64_t peak_rss_mib);

// Aggregate DiT graph-load lines, e.g. `profile name=dit-param-load ms=123.4`.
std::string format_profile_dit_load_line(const char* component, double elapsed_ms);

// Aggregate ncnn model-load stages for the DiT stack, e.g.
// `profile name=dit-ncnn-upload-submit ms=123.4`.
std::string format_profile_dit_stage_line(const char* stage, double elapsed_ms);
std::string format_profile_runtime_line(std::size_t graph_loads,
                                        std::size_t transient_graph_loads,
                                        double graph_load_ms,
                                        double transient_graph_load_ms,
                                        std::size_t submits,
                                        double submit_ms,
                                        std::size_t uploads,
                                        std::size_t downloads);

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

    void report_mode(const char* name, const char* mode) const
    {
        if (enabled_)
            std::fprintf(stderr, "%s\n", format_profile_mode_line(name, mode).c_str());
    }

    void report_residency(const char* phase,
                          std::uint32_t heap_budget_mib,
                          std::uint64_t max_allocation_mib) const
    {
        if (enabled_)
        {
            std::fprintf(stderr, "%s\n",
                         format_profile_residency_line(phase, current_rss_mib(), peak_rss_mib(),
                                                       heap_budget_mib, max_allocation_mib)
                             .c_str());
        }
    }

    void report_session_open(double elapsed_ms) const
    {
        if (enabled_)
            std::fprintf(stderr, "%s\n", format_profile_session_open_line("cold", elapsed_ms).c_str());
    }

    void report_session_run(std::size_t run_index, double elapsed_ms) const
    {
        if (enabled_)
        {
            const char* mode = run_index == 0 ? "cold" : "warm";
            std::fprintf(stderr, "%s\n",
                         format_profile_session_run_line(mode, run_index, elapsed_ms).c_str());
        }
    }

    void report_pipeline_cache(const char* phase, std::size_t entries) const
    {
        if (enabled_)
            std::fprintf(stderr, "%s\n", format_profile_pipeline_cache_line(phase, entries).c_str());
    }

    void report_model_cache(const char* phase, const char* mode) const
    {
        if (enabled_)
            std::fprintf(stderr, "%s\n", format_profile_model_cache_line(phase, mode).c_str());
    }

    void report_total(double elapsed_ms) const
    {
        if (enabled_)
        {
            std::fprintf(stderr,
                         "%s\n",
                         format_profile_runtime_line(runtime_graph_loads_, runtime_transient_graph_loads_,
                                                     runtime_graph_load_ms_, runtime_transient_graph_load_ms_,
                                                     runtime_submits_, runtime_submit_ms_, runtime_uploads_,
                                                     runtime_downloads_)
                             .c_str());
            std::fprintf(stderr, "%s\n",
                         format_profile_total_line(elapsed_ms, peak_rss_mib()).c_str());
        }
    }

    void record_runtime_graph_load(bool transient, double elapsed_ms) const
    {
        if (!enabled_)
            return;
        ++runtime_graph_loads_;
        runtime_graph_load_ms_ += elapsed_ms;
        if (transient)
        {
            ++runtime_transient_graph_loads_;
            runtime_transient_graph_load_ms_ += elapsed_ms;
        }
    }

    void record_runtime_submit(double elapsed_ms) const
    {
        if (enabled_)
        {
            ++runtime_submits_;
            runtime_submit_ms_ += elapsed_ms;
        }
    }

    void record_runtime_upload() const
    {
        if (enabled_)
            ++runtime_uploads_;
    }

    void record_runtime_download() const
    {
        if (enabled_)
            ++runtime_downloads_;
    }

    // Peak resident host memory in MiB, or 0 when the platform cannot report it.
    std::uint64_t peak_rss_mib() const;

    // Current resident host memory in MiB, or 0 when the platform cannot report it.
    std::uint64_t current_rss_mib() const;

private:
    bool enabled_ = false;
    mutable std::size_t runtime_graph_loads_ = 0;
    mutable std::size_t runtime_transient_graph_loads_ = 0;
    mutable double runtime_graph_load_ms_ = 0.0;
    mutable double runtime_transient_graph_load_ms_ = 0.0;
    mutable std::size_t runtime_submits_ = 0;
    mutable double runtime_submit_ms_ = 0.0;
    mutable std::size_t runtime_uploads_ = 0;
    mutable std::size_t runtime_downloads_ = 0;
};

// RAII helper that reports the elapsed time of a scope on destruction.
class ProfileScope final
{
public:
    ProfileScope(const PerformanceProfile& profile, const char* name)
        : profile_(profile), name_(name),
          start_(profile.enabled() ? PerformanceProfile::Clock::now() : PerformanceProfile::Clock::time_point{})
    {
    }

    ProfileScope(const PerformanceProfile& profile, const char* name, std::size_t frame_index)
        : ProfileScope(profile, name)
    {
        frame_index_ = frame_index;
        has_frame_ = true;
    }

    ~ProfileScope()
    {
        if (!profile_.enabled())
            return;
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
