#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "io/image_io.h"

namespace seedvr2
{

struct VideoInfo final
{
    int width = 0;
    int height = 0;
    int frame_count = 0;
    int fps_num = 0;
    int fps_den = 1;
};

class AviVideoReader final
{
public:
    AviVideoReader() = default;
    ~AviVideoReader();

    static bool open(const std::filesystem::path& path, AviVideoReader& reader, std::string& error);

    const VideoInfo& info() const { return info_; }
    bool read_next(RgbImage& frame, std::string& error);

private:
    std::filesystem::path path_;
    std::FILE* file_ = nullptr;
    VideoInfo info_;
    std::uint64_t frame_data_offset_ = 0;
    std::uint32_t frame_data_size_ = 0;
    std::uint64_t next_chunk_offset_ = 0;
    std::uint64_t movi_end_offset_ = 0;
    std::uint32_t row_stride_ = 0;
    bool top_down_ = false;
    int frame_index_ = 0;
};

class AviVideoWriter final
{
public:
    AviVideoWriter() = default;
    ~AviVideoWriter();

    static bool open(const std::filesystem::path& path, const VideoInfo& info, AviVideoWriter& writer,
                     std::string& error);

    bool write_frame(const RgbImage& frame, std::string& error);
    bool close(std::string& error);

private:
    std::filesystem::path path_;
    std::FILE* file_ = nullptr;
    VideoInfo info_;
    std::uint64_t movi_offset_ = 0;
    std::uint64_t riff_size_offset_ = 0;
    std::uint64_t avih_frames_offset_ = 0;
    std::uint64_t strh_frames_offset_ = 0;
    std::uint64_t movi_size_offset_ = 0;
    std::uint64_t movi_data_offset_ = 0;
    std::uint32_t frame_count_ = 0;
};

class VideoReader final
{
public:
    VideoReader();
    ~VideoReader();

    VideoReader(const VideoReader&) = delete;
    VideoReader& operator=(const VideoReader&) = delete;
    VideoReader(VideoReader&&) noexcept;
    VideoReader& operator=(VideoReader&&) noexcept;

    static bool open(const std::filesystem::path& path, VideoReader& reader, std::string& error);

    const VideoInfo& info() const;
    bool read_next(RgbImage& frame, std::string& error);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace seedvr2
