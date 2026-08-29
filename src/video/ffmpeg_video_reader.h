#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "video/video_io.h"

namespace seedvr2
{

class FfmpegVideoReader final
{
public:
    FfmpegVideoReader();
    ~FfmpegVideoReader();

    FfmpegVideoReader(const FfmpegVideoReader&) = delete;
    FfmpegVideoReader& operator=(const FfmpegVideoReader&) = delete;
    FfmpegVideoReader(FfmpegVideoReader&&) noexcept;
    FfmpegVideoReader& operator=(FfmpegVideoReader&&) noexcept;

    static bool open(const std::filesystem::path& path, FfmpegVideoReader& reader, std::string& error);

    const VideoInfo& info() const;
    bool read_next(RgbImage& frame, std::string& error);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace seedvr2
