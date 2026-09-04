#include "inference/rgb_spool.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

namespace seedvr2
{
namespace
{

constexpr std::array<char, 8> kMagic = {'S', 'V', 'R', '2', 'R', 'G', 'B', '1'};
constexpr std::uint32_t kHeaderBytes = 20;
constexpr std::uint64_t kMaxBytes = 1ull << 30;

void write_u32(unsigned char* data, std::uint32_t value)
{
    data[0] = static_cast<unsigned char>(value & 0xffu);
    data[1] = static_cast<unsigned char>((value >> 8) & 0xffu);
    data[2] = static_cast<unsigned char>((value >> 16) & 0xffu);
    data[3] = static_cast<unsigned char>((value >> 24) & 0xffu);
}

std::uint32_t read_u32(const unsigned char* data)
{
    return static_cast<std::uint32_t>(data[0]) |
           (static_cast<std::uint32_t>(data[1]) << 8) |
           (static_cast<std::uint32_t>(data[2]) << 16) |
           (static_cast<std::uint32_t>(data[3]) << 24);
}

bool valid_frame(const RgbImage& frame, std::uint64_t& bytes)
{
    if (frame.width <= 0 || frame.height <= 0 ||
        static_cast<std::uint64_t>(frame.width) > std::numeric_limits<std::uint32_t>::max() ||
        static_cast<std::uint64_t>(frame.height) > std::numeric_limits<std::uint32_t>::max())
        return false;
    bytes = static_cast<std::uint64_t>(frame.width) * frame.height * 3u;
    return bytes > 0 && bytes <= kMaxBytes && frame.pixels.size() == bytes;
}

} // namespace

RgbFrameSpool::~RgbFrameSpool()
{
    close();
}

RgbFrameSpool::RgbFrameSpool(RgbFrameSpool&& other) noexcept : file_(other.file_), writing_(other.writing_)
{
    other.file_ = nullptr;
}

RgbFrameSpool& RgbFrameSpool::operator=(RgbFrameSpool&& other) noexcept
{
    if (this == &other)
        return *this;
    close();
    file_ = other.file_;
    writing_ = other.writing_;
    other.file_ = nullptr;
    return *this;
}

bool RgbFrameSpool::create(RgbFrameSpool& spool, std::string& error)
{
    error.clear();
    spool.close();
    std::FILE* file = std::tmpfile();
    if (!file)
    {
        error = "failed to create a temporary RGB reference spool";
        return false;
    }
    spool.file_ = file;
    spool.writing_ = true;
    return true;
}

void RgbFrameSpool::close()
{
    if (file_)
    {
        std::fclose(file_);
        file_ = nullptr;
    }
}

bool RgbFrameSpool::append(const RgbImage& frame, std::string& error)
{
    error.clear();
    std::uint64_t bytes = 0;
    if (!file_ || !writing_ || !valid_frame(frame, bytes))
    {
        error = "RGB reference spool frame is invalid or not writable";
        return false;
    }
    unsigned char header[kHeaderBytes];
    std::memcpy(header, kMagic.data(), kMagic.size());
    write_u32(header + 8, static_cast<std::uint32_t>(frame.width));
    write_u32(header + 12, static_cast<std::uint32_t>(frame.height));
    write_u32(header + 16, 3);
    if (std::fwrite(header, 1, sizeof(header), file_) != sizeof(header) ||
        std::fwrite(frame.pixels.data(), 1, static_cast<std::size_t>(bytes), file_) != bytes)
    {
        error = "failed to append RGB reference frame to spool";
        return false;
    }
    return true;
}

bool RgbFrameSpool::rewind(std::string& error)
{
    error.clear();
    if (!file_ || std::fflush(file_) != 0 || std::fseek(file_, 0, SEEK_SET) != 0)
    {
        error = "failed to rewind RGB reference spool";
        return false;
    }
    writing_ = false;
    return true;
}

bool RgbFrameSpool::read_next(RgbImage& frame, std::string& error)
{
    frame = RgbImage();
    error.clear();
    if (!file_ || writing_)
    {
        error = "RGB reference spool is not readable";
        return false;
    }
    unsigned char header[kHeaderBytes];
    const std::size_t first = std::fread(header, 1, sizeof(header), file_);
    if (first == 0 && std::feof(file_))
        return false;
    if (first != sizeof(header) || std::memcmp(header, kMagic.data(), kMagic.size()) != 0 || read_u32(header + 16) != 3)
    {
        error = "RGB reference spool record header is truncated or invalid";
        return false;
    }
    const std::uint32_t width = read_u32(header + 8);
    const std::uint32_t height = read_u32(header + 12);
    const std::uint64_t bytes = static_cast<std::uint64_t>(width) * height * 3u;
    if (width == 0 || height == 0 || bytes == 0 || bytes > kMaxBytes ||
        width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        height > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
    {
        error = "RGB reference spool record dimensions are invalid";
        return false;
    }
    frame.width = static_cast<int>(width);
    frame.height = static_cast<int>(height);
    frame.pixels.resize(static_cast<std::size_t>(bytes));
    if (std::fread(frame.pixels.data(), 1, frame.pixels.size(), file_) != frame.pixels.size())
    {
        frame = RgbImage();
        error = "RGB reference spool record payload is truncated";
        return false;
    }
    return true;
}

} // namespace seedvr2
