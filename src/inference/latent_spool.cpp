#include "inference/latent_spool.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

namespace seedvr2
{
namespace
{

constexpr std::array<char, 8> kMagic = {'S', 'V', 'R', '2', 'L', 'T', 'S', '1'};
constexpr std::uint32_t kHeaderBytes = 20;
constexpr std::uint32_t kMaxElements = 1u << 28;

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

bool valid_frame(const LatentFrame& frame, std::uint32_t& elements)
{
    if (frame.width <= 0 || frame.height <= 0 || frame.channels <= 0)
        return false;
    const std::uint64_t count = static_cast<std::uint64_t>(frame.width) *
                                static_cast<std::uint64_t>(frame.height) *
                                static_cast<std::uint64_t>(frame.channels);
    if (count == 0 || count > kMaxElements || count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return false;
    elements = static_cast<std::uint32_t>(count);
    return frame.values.size() == count;
}

} // namespace

LatentSpool::~LatentSpool()
{
    close();
}

LatentSpool::LatentSpool(LatentSpool&& other) noexcept : file_(other.file_), writing_(other.writing_)
{
    other.file_ = nullptr;
}

LatentSpool& LatentSpool::operator=(LatentSpool&& other) noexcept
{
    if (this == &other)
        return *this;
    close();
    file_ = other.file_;
    writing_ = other.writing_;
    other.file_ = nullptr;
    return *this;
}

bool LatentSpool::create(LatentSpool& spool, std::string& error)
{
    error.clear();
    spool.close();
    std::FILE* file = std::tmpfile();
    if (!file)
    {
        error = "failed to create a temporary latent spool";
        return false;
    }
    spool.file_ = file;
    spool.writing_ = true;
    return true;
}

void LatentSpool::close()
{
    if (file_)
    {
        std::fclose(file_);
        file_ = nullptr;
    }
}

bool LatentSpool::append(const LatentFrame& frame, std::string& error)
{
    error.clear();
    std::uint32_t elements = 0;
    if (!file_ || !writing_ || !valid_frame(frame, elements))
    {
        error = "latent spool frame is invalid or not writable";
        return false;
    }

    unsigned char header[kHeaderBytes];
    std::memcpy(header, kMagic.data(), kMagic.size());
    write_u32(header + 8, static_cast<std::uint32_t>(frame.width));
    write_u32(header + 12, static_cast<std::uint32_t>(frame.height));
    write_u32(header + 16, static_cast<std::uint32_t>(frame.channels));
    if (std::fwrite(header, 1, sizeof(header), file_) != sizeof(header) ||
        std::fwrite(frame.values.data(), sizeof(float), elements, file_) != elements)
    {
        error = "failed to append latent frame to spool";
        return false;
    }
    return true;
}

bool LatentSpool::rewind(std::string& error)
{
    error.clear();
    if (!file_ || std::fflush(file_) != 0 || std::fseek(file_, 0, SEEK_SET) != 0)
    {
        error = "failed to rewind latent spool";
        return false;
    }
    writing_ = false;
    return true;
}

bool LatentSpool::read_next(LatentFrame& frame, std::string& error)
{
    frame = LatentFrame();
    error.clear();
    if (!file_ || writing_)
    {
        error = "latent spool is not readable";
        return false;
    }

    unsigned char header[kHeaderBytes];
    const std::size_t first = std::fread(header, 1, sizeof(header), file_);
    if (first == 0 && std::feof(file_))
        return false;
    if (first != sizeof(header) || std::memcmp(header, kMagic.data(), kMagic.size()) != 0)
    {
        error = "latent spool record header is truncated or invalid";
        return false;
    }

    const std::uint32_t width = read_u32(header + 8);
    const std::uint32_t height = read_u32(header + 12);
    const std::uint32_t channels = read_u32(header + 16);
    if (width == 0 || width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        height == 0 || height > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        channels == 0 || channels > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
    {
        error = "latent spool record dimensions are invalid";
        return false;
    }
    const std::uint64_t count = static_cast<std::uint64_t>(width) * height * channels;
    if (count == 0 || count > kMaxElements || count > std::numeric_limits<std::size_t>::max() / sizeof(float))
    {
        error = "latent spool record is too large";
        return false;
    }

    frame.width = static_cast<int>(width);
    frame.height = static_cast<int>(height);
    frame.channels = static_cast<int>(channels);
    frame.values.resize(static_cast<std::size_t>(count));
    if (std::fread(frame.values.data(), sizeof(float), frame.values.size(), file_) != frame.values.size())
    {
        frame = LatentFrame();
        error = "latent spool record payload is truncated";
        return false;
    }
    return true;
}

} // namespace seedvr2
