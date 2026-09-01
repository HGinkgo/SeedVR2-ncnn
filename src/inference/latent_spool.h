#pragma once

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace seedvr2
{

struct LatentFrame final
{
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<float> values;
};

// Private, process-local storage used to connect sequential Vulkan phases
// without keeping all video latents resident at once.
class LatentSpool final
{
public:
    LatentSpool() = default;
    ~LatentSpool();

    LatentSpool(LatentSpool&& other) noexcept;
    LatentSpool& operator=(LatentSpool&& other) noexcept;
    LatentSpool(const LatentSpool&) = delete;
    LatentSpool& operator=(const LatentSpool&) = delete;

    static bool create(LatentSpool& spool, std::string& error);

    bool append(const LatentFrame& frame, std::string& error);
    bool rewind(std::string& error);
    // Returns false with an empty error only at clean end-of-file.
    bool read_next(LatentFrame& frame, std::string& error);

private:
    explicit LatentSpool(std::FILE* file) : file_(file) {}
    void close();

    std::FILE* file_ = nullptr;
    bool writing_ = true;
};

} // namespace seedvr2
