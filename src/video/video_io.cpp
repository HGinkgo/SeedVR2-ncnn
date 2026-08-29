#include "video/video_io.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace seedvr2
{
namespace
{

bool has_avi_extension(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    for (char& value : extension)
        if (value >= 'A' && value <= 'Z')
            value = static_cast<char>(value - 'A' + 'a');
    return extension == ".avi";
}

bool read_exact(std::FILE* file, void* data, std::size_t size)
{
    return size == 0 || std::fread(data, 1, size, file) == size;
}

bool write_exact(std::FILE* file, const void* data, std::size_t size)
{
    return size == 0 || std::fwrite(data, 1, size, file) == size;
}

std::uint32_t read_u32(const unsigned char* data)
{
    return static_cast<std::uint32_t>(data[0]) |
           (static_cast<std::uint32_t>(data[1]) << 8) |
           (static_cast<std::uint32_t>(data[2]) << 16) |
           (static_cast<std::uint32_t>(data[3]) << 24);
}

std::uint16_t read_u16(const unsigned char* data)
{
    return static_cast<std::uint16_t>(data[0]) |
           static_cast<std::uint16_t>(data[1] << 8);
}

void write_u32(unsigned char* data, std::uint32_t value)
{
    data[0] = static_cast<unsigned char>(value & 0xffu);
    data[1] = static_cast<unsigned char>((value >> 8) & 0xffu);
    data[2] = static_cast<unsigned char>((value >> 16) & 0xffu);
    data[3] = static_cast<unsigned char>((value >> 24) & 0xffu);
}

void write_u16(unsigned char* data, std::uint16_t value)
{
    data[0] = static_cast<unsigned char>(value & 0xffu);
    data[1] = static_cast<unsigned char>((value >> 8) & 0xffu);
}

bool fourcc_equals(const unsigned char* value, const char* expected)
{
    return std::memcmp(value, expected, 4) == 0;
}

std::uint64_t file_offset(std::FILE* file)
{
    const long offset = std::ftell(file);
    return offset < 0 ? 0 : static_cast<std::uint64_t>(offset);
}

bool seek_to(std::FILE* file, std::uint64_t offset)
{
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<long>::max()))
        return false;
    return std::fseek(file, static_cast<long>(offset), SEEK_SET) == 0;
}

bool skip_chunk_padding(std::FILE* file, std::uint32_t size)
{
    return (size & 1u) == 0 || std::fseek(file, 1, SEEK_CUR) == 0;
}

bool parse_header_list(std::FILE* file, std::uint64_t begin, std::uint64_t end, VideoInfo& info,
                       std::uint32_t& compression, std::uint16_t& planes, std::uint16_t& bit_count)
{
    if (!seek_to(file, begin))
        return false;
    while (file_offset(file) + 8 <= end)
    {
        const std::uint64_t chunk_start = file_offset(file);
        unsigned char chunk_header[8];
        if (!read_exact(file, chunk_header, sizeof(chunk_header)))
            return false;
        const std::uint32_t size = read_u32(chunk_header + 4);
        const std::uint64_t payload = file_offset(file);
        const std::uint64_t chunk_end = payload + size;
        if (chunk_end > end)
            return false;

        if (fourcc_equals(chunk_header, "LIST"))
        {
            if (size < 4)
                return false;
            unsigned char list_type[4];
            if (!read_exact(file, list_type, sizeof(list_type)))
                return false;
            if (!parse_header_list(file, file_offset(file), chunk_end, info, compression, planes, bit_count))
                return false;
        }
        else if (fourcc_equals(chunk_header, "avih"))
        {
            std::array<unsigned char, 56> data{};
            const std::size_t to_read = std::min<std::size_t>(data.size(), size);
            if (!read_exact(file, data.data(), to_read))
                return false;
            if (to_read >= data.size())
            {
                const std::uint32_t microseconds = read_u32(data.data());
                info.frame_count = static_cast<int>(read_u32(data.data() + 16));
                info.width = static_cast<int>(read_u32(data.data() + 32));
                info.height = static_cast<int>(read_u32(data.data() + 36));
                if (microseconds > 0)
                {
                    info.fps_num = 1000000;
                    info.fps_den = static_cast<int>(microseconds);
                }
            }
        }
        else if (fourcc_equals(chunk_header, "strh"))
        {
            std::array<unsigned char, 56> data{};
            const std::size_t to_read = std::min<std::size_t>(data.size(), size);
            if (!read_exact(file, data.data(), to_read))
                return false;
            if (to_read >= data.size() && fourcc_equals(data.data(), "vids"))
            {
                const std::uint32_t scale = read_u32(data.data() + 20);
                const std::uint32_t rate = read_u32(data.data() + 24);
                if (scale > 0 && rate > 0)
                {
                    info.fps_num = static_cast<int>(rate);
                    info.fps_den = static_cast<int>(scale);
                }
                if (info.frame_count == 0)
                    info.frame_count = static_cast<int>(read_u32(data.data() + 32));
            }
        }
        else if (fourcc_equals(chunk_header, "strf"))
        {
            std::array<unsigned char, 40> data{};
            const std::size_t to_read = std::min<std::size_t>(data.size(), size);
            if (!read_exact(file, data.data(), to_read))
                return false;
            if (to_read >= data.size())
            {
                info.width = static_cast<int>(read_u32(data.data() + 4));
                info.height = static_cast<int>(read_u32(data.data() + 8));
                planes = read_u16(data.data() + 12);
                bit_count = read_u16(data.data() + 14);
                compression = read_u32(data.data() + 16);
            }
        }

        if (!seek_to(file, chunk_end) || !skip_chunk_padding(file, size))
            return false;
        (void)chunk_start;
    }
    return file_offset(file) == end;
}

bool write_chunk_header(std::FILE* file, const char* id, std::uint32_t size)
{
    unsigned char header[8];
    std::memcpy(header, id, 4);
    write_u32(header + 4, size);
    return write_exact(file, header, sizeof(header));
}

bool patch_u32(std::FILE* file, std::uint64_t offset, std::uint32_t value)
{
    if (!seek_to(file, offset))
        return false;
    unsigned char data[4];
    write_u32(data, value);
    return write_exact(file, data, sizeof(data));
}

} // namespace

class VideoReader::Impl final
{
public:
    enum class Backend
    {
        Avi,
    };

    Backend backend = Backend::Avi;
    AviVideoReader avi;
};

VideoReader::~VideoReader() = default;

VideoReader::VideoReader() = default;

VideoReader::VideoReader(VideoReader&&) noexcept = default;

VideoReader& VideoReader::operator=(VideoReader&&) noexcept = default;

bool VideoReader::open(const std::filesystem::path& path, VideoReader& reader, std::string& error)
{
    reader.impl_.reset();
    error.clear();
    if (!has_avi_extension(path))
    {
        error = "compressed video input requires a build with SEEDVR2_ENABLE_FFMPEG=ON";
        return false;
    }

    auto implementation = std::make_unique<Impl>();
    if (!AviVideoReader::open(path, implementation->avi, error))
        return false;
    reader.impl_ = std::move(implementation);
    return true;
}

const VideoInfo& VideoReader::info() const
{
    static const VideoInfo empty_info;
    return impl_ ? impl_->avi.info() : empty_info;
}

bool VideoReader::read_next(RgbImage& frame, std::string& error)
{
    if (!impl_)
    {
        frame = RgbImage();
        error = "video reader is not open";
        return false;
    }
    return impl_->avi.read_next(frame, error);
}

AviVideoReader::~AviVideoReader()
{
    if (file_)
        std::fclose(file_);
}

bool AviVideoReader::open(const std::filesystem::path& path, AviVideoReader& reader, std::string& error)
{
    if (reader.file_)
        std::fclose(reader.file_);
    reader = AviVideoReader();
    error.clear();
    std::FILE* file = std::fopen(path.string().c_str(), "rb");
    if (!file)
    {
        error = "failed to open AVI input: " + path.string();
        return false;
    }

    unsigned char riff_header[12];
    if (!read_exact(file, riff_header, sizeof(riff_header)) || !fourcc_equals(riff_header, "RIFF") ||
        !fourcc_equals(riff_header + 8, "AVI "))
    {
        std::fclose(file);
        error = "input is not a RIFF AVI video";
        return false;
    }

    const std::uint64_t file_end = 8u + read_u32(riff_header + 4);
    VideoInfo info;
    std::uint32_t compression = 0;
    std::uint16_t planes = 0;
    std::uint16_t bit_count = 0;
    std::uint64_t movi_data = 0;
    std::uint64_t movi_end = 0;
    while (file_offset(file) + 8 <= file_end)
    {
        unsigned char chunk_header[8];
        if (!read_exact(file, chunk_header, sizeof(chunk_header)))
            break;
        const std::uint32_t size = read_u32(chunk_header + 4);
        const std::uint64_t chunk_end = file_offset(file) + size;
        if (chunk_end > file_end)
            break;
        if (fourcc_equals(chunk_header, "LIST"))
        {
            if (size < 4)
                break;
            unsigned char list_type[4];
            if (!read_exact(file, list_type, sizeof(list_type)))
                break;
            if (fourcc_equals(list_type, "hdrl"))
            {
                if (!parse_header_list(file, file_offset(file), chunk_end, info, compression, planes, bit_count))
                    break;
            }
            else if (fourcc_equals(list_type, "movi"))
            {
                movi_data = file_offset(file);
                movi_end = chunk_end;
            }
        }
        if (!seek_to(file, chunk_end) || !skip_chunk_padding(file, size))
            break;
    }

    if (movi_data == 0 || movi_end <= movi_data || info.width <= 0 || info.height <= 0 || planes != 1 ||
        bit_count != 24 || compression != 0 || info.fps_num <= 0 || info.fps_den <= 0)
    {
        std::fclose(file);
        error = "AVI must contain an uncompressed 24-bit video stream";
        return false;
    }

    reader.path_ = path;
    reader.file_ = file;
    reader.info_ = info;
    reader.top_down_ = reader.info_.height < 0;
    reader.info_.height = std::abs(reader.info_.height);
    reader.next_chunk_offset_ = movi_data;
    reader.movi_end_offset_ = movi_end;
    reader.row_stride_ = (static_cast<std::uint32_t>(reader.info_.width) * 3u + 3u) & ~3u;
    reader.frame_index_ = 0;
    return true;
}

bool AviVideoReader::read_next(RgbImage& frame, std::string& error)
{
    frame = RgbImage();
    error.clear();
    if (!file_)
    {
        error = "AVI reader is not open";
        return false;
    }

    while (next_chunk_offset_ + 8 <= movi_end_offset_)
    {
        if (!seek_to(file_, next_chunk_offset_))
        {
            error = "failed to seek AVI frame";
            return false;
        }
        unsigned char header[8];
        if (!read_exact(file_, header, sizeof(header)))
        {
            error = "failed to read AVI frame header";
            return false;
        }
        const std::uint32_t size = read_u32(header + 4);
        const std::uint64_t payload = file_offset(file_);
        const std::uint64_t chunk_end = payload + size;
        if (chunk_end > movi_end_offset_)
        {
            error = "AVI frame exceeds the media chunk";
            return false;
        }
        next_chunk_offset_ = chunk_end + (size & 1u);

        if (!(header[0] >= '0' && header[0] <= '9' && header[1] >= '0' && header[1] <= '9' &&
              (header[2] == 'd' || header[2] == 'D') && (header[3] == 'b' || header[3] == 'B')))
            continue;

        const std::uint64_t expected = static_cast<std::uint64_t>(row_stride_) * info_.height;
        if (size < expected || expected > std::numeric_limits<std::uint32_t>::max())
        {
            error = "AVI video frame is not an uncompressed 24-bit image";
            return false;
        }
        std::vector<unsigned char> bgr(static_cast<std::size_t>(expected));
        if (!seek_to(file_, payload) || !read_exact(file_, bgr.data(), bgr.size()))
        {
            error = "failed to read AVI video frame";
            return false;
        }

        frame.width = info_.width;
        frame.height = info_.height;
        frame.pixels.resize(static_cast<std::size_t>(frame.width) * frame.height * 3u);
        for (int y = 0; y < frame.height; y++)
        {
            const int source_y = top_down_ ? y : info_.height - 1 - y;
            const unsigned char* source = bgr.data() + static_cast<std::size_t>(source_y) * row_stride_;
            unsigned char* destination = frame.pixels.data() + static_cast<std::size_t>(y) * frame.width * 3u;
            for (int x = 0; x < frame.width; x++)
            {
                destination[x * 3 + 0] = source[x * 3 + 2];
                destination[x * 3 + 1] = source[x * 3 + 1];
                destination[x * 3 + 2] = source[x * 3 + 0];
            }
        }
        frame_index_++;
        return true;
    }
    return false;
}

AviVideoWriter::~AviVideoWriter()
{
    if (file_)
    {
        std::string ignored;
        close(ignored);
    }
}

bool AviVideoWriter::open(const std::filesystem::path& path, const VideoInfo& info, AviVideoWriter& writer,
                          std::string& error)
{
    if (writer.file_)
    {
        std::string ignored;
        writer.close(ignored);
    }
    writer = AviVideoWriter();
    error.clear();
    if (info.width <= 0 || info.height <= 0 || info.fps_num <= 0 || info.fps_den <= 0)
    {
        error = "AVI output dimensions and frame rate must be positive";
        return false;
    }
    const std::uint64_t row_stride = (static_cast<std::uint64_t>(info.width) * 3u + 3u) & ~3u;
    const std::uint64_t frame_bytes = row_stride * info.height;
    if (frame_bytes > std::numeric_limits<std::uint32_t>::max())
    {
        error = "AVI frame is too large for the uncompressed writer";
        return false;
    }
    std::FILE* file = std::fopen(path.string().c_str(), "wb+");
    if (!file)
    {
        error = "failed to open AVI output: " + path.string();
        return false;
    }

    const std::uint32_t microseconds = static_cast<std::uint32_t>(
        (1000000ull * static_cast<std::uint64_t>(info.fps_den)) / static_cast<std::uint64_t>(info.fps_num));
    const std::uint32_t bytes_per_second = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(std::numeric_limits<std::uint32_t>::max(), frame_bytes * info.fps_num / info.fps_den));
    unsigned char zeros[56] = {};
    unsigned char avih[56] = {};
    write_u32(avih, microseconds);
    write_u32(avih + 4, bytes_per_second);
    write_u32(avih + 12, 0x10u);
    write_u32(avih + 24, 1u);
    write_u32(avih + 28, static_cast<std::uint32_t>(frame_bytes));
    write_u32(avih + 32, static_cast<std::uint32_t>(info.width));
    write_u32(avih + 36, static_cast<std::uint32_t>(info.height));
    unsigned char strh[56] = {};
    std::memcpy(strh, "vids", 4);
    std::memcpy(strh + 4, "DIB ", 4);
    write_u32(strh + 20, static_cast<std::uint32_t>(info.fps_den));
    write_u32(strh + 24, static_cast<std::uint32_t>(info.fps_num));
    write_u32(strh + 36, static_cast<std::uint32_t>(frame_bytes));
    write_u32(strh + 40, 0xffffffffu);
    unsigned char strf[40] = {};
    write_u32(strf, 40u);
    write_u32(strf + 4, static_cast<std::uint32_t>(info.width));
    write_u32(strf + 8, static_cast<std::uint32_t>(info.height));
    write_u16(strf + 12, 1u);
    write_u16(strf + 14, 24u);
    write_u32(strf + 20, static_cast<std::uint32_t>(frame_bytes));
    write_u32(strf + 24, 2835u);
    write_u32(strf + 28, 2835u);

    const auto fail = [&]() {
        std::fclose(file);
        error = "failed to write AVI header: " + path.string();
        return false;
    };
    if (!write_exact(file, "RIFF", 4))
        return fail();
    writer.riff_size_offset_ = file_offset(file);
    if (!write_exact(file, zeros, 4) || !write_exact(file, "AVI ", 4) || !write_exact(file, "LIST", 4))
        return fail();
    const std::uint64_t hdrl_size_offset = file_offset(file);
    if (!write_exact(file, zeros, 4) || !write_exact(file, "hdrl", 4) || !write_chunk_header(file, "avih", 56))
        return fail();
    writer.avih_frames_offset_ = file_offset(file) + 16;
    if (!write_exact(file, avih, sizeof(avih)) || !write_exact(file, "LIST", 4))
        return fail();
    const std::uint64_t strl_size_offset = file_offset(file);
    if (!write_exact(file, zeros, 4) || !write_exact(file, "strl", 4) || !write_chunk_header(file, "strh", 56))
        return fail();
    writer.strh_frames_offset_ = file_offset(file) + 32;
    if (!write_exact(file, strh, sizeof(strh)) || !write_chunk_header(file, "strf", 40) ||
        !write_exact(file, strf, sizeof(strf)))
        return fail();
    {
        const std::uint64_t strl_end = file_offset(file);
        if (!patch_u32(file, strl_size_offset, static_cast<std::uint32_t>(strl_end - strl_size_offset - 4)))
            return fail();
        if (!seek_to(file, strl_end))
            return fail();
    }
    {
        const std::uint64_t hdrl_end = file_offset(file);
        if (!patch_u32(file, hdrl_size_offset, static_cast<std::uint32_t>(hdrl_end - hdrl_size_offset - 4)))
            return fail();
        if (!seek_to(file, hdrl_end))
            return fail();
    }
    if (!write_exact(file, "LIST", 4))
        return fail();
    writer.movi_size_offset_ = file_offset(file);
    if (!write_exact(file, zeros, 4) || !write_exact(file, "movi", 4))
        return fail();
    writer.movi_data_offset_ = file_offset(file);
    writer.path_ = path;
    writer.file_ = file;
    writer.info_ = info;
    return true;
}

bool AviVideoWriter::write_frame(const RgbImage& frame, std::string& error)
{
    error.clear();
    if (!file_ || frame.width != info_.width || frame.height != info_.height ||
        frame.pixels.size() != static_cast<std::size_t>(frame.width) * frame.height * 3u)
    {
        error = "AVI frame dimensions do not match the output video";
        return false;
    }
    const std::uint32_t row_stride = (static_cast<std::uint32_t>(info_.width) * 3u + 3u) & ~3u;
    const std::uint32_t frame_bytes = row_stride * static_cast<std::uint32_t>(info_.height);
    if (!write_chunk_header(file_, "00db", frame_bytes))
    {
        error = "failed to write AVI frame header";
        return false;
    }
    std::vector<unsigned char> row(row_stride, 0);
    for (int y = info_.height - 1; y >= 0; y--)
    {
        const unsigned char* source = frame.pixels.data() + static_cast<std::size_t>(y) * info_.width * 3u;
        for (int x = 0; x < info_.width; x++)
        {
            row[x * 3 + 0] = source[x * 3 + 2];
            row[x * 3 + 1] = source[x * 3 + 1];
            row[x * 3 + 2] = source[x * 3 + 0];
        }
        if (!write_exact(file_, row.data(), row.size()))
        {
            error = "failed to write AVI frame data";
            return false;
        }
    }
    if (frame_bytes & 1u)
    {
        const unsigned char pad = 0;
        if (!write_exact(file_, &pad, 1))
        {
            error = "failed to write AVI frame padding";
            return false;
        }
    }
    if (frame_count_ == std::numeric_limits<std::uint32_t>::max())
    {
        error = "AVI frame count exceeds the supported limit";
        return false;
    }
    frame_count_++;
    return true;
}

bool AviVideoWriter::close(std::string& error)
{
    error.clear();
    if (!file_)
        return true;
    const std::uint64_t end = file_offset(file_);
    const std::uint64_t movi_size = end - movi_data_offset_ + 4;
    const std::uint64_t riff_size = end - 8;
    const bool sizes_fit = movi_size <= std::numeric_limits<std::uint32_t>::max() &&
                           riff_size <= std::numeric_limits<std::uint32_t>::max();
    const bool patched = sizes_fit && patch_u32(file_, riff_size_offset_, static_cast<std::uint32_t>(riff_size)) &&
                         patch_u32(file_, avih_frames_offset_, frame_count_) &&
                         patch_u32(file_, strh_frames_offset_, frame_count_) &&
                         patch_u32(file_, movi_size_offset_, static_cast<std::uint32_t>(movi_size)) &&
                         std::fflush(file_) == 0;
    std::fclose(file_);
    file_ = nullptr;
    if (!patched)
    {
        error = "failed to finalize AVI output";
        return false;
    }
    return true;
}

} // namespace seedvr2
