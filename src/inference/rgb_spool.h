#pragma once

#include <cstdio>
#include <string>

#include "io/image_io.h"

namespace seedvr2
{

// Private process-local storage for video RGB references used by optional postprocessing.
class RgbFrameSpool final
{
public:
    RgbFrameSpool() = default;
    ~RgbFrameSpool();

    RgbFrameSpool(RgbFrameSpool&& other) noexcept;
    RgbFrameSpool& operator=(RgbFrameSpool&& other) noexcept;
    RgbFrameSpool(const RgbFrameSpool&) = delete;
    RgbFrameSpool& operator=(const RgbFrameSpool&) = delete;

    static bool create(RgbFrameSpool& spool, std::string& error);
    bool append(const RgbImage& frame, std::string& error);
    bool rewind(std::string& error);
    bool read_next(RgbImage& frame, std::string& error);

private:
    explicit RgbFrameSpool(std::FILE* file) : file_(file) {}
    void close();

    std::FILE* file_ = nullptr;
    bool writing_ = true;
};

} // namespace seedvr2
