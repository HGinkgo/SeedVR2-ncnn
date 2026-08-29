#include "video/video_io.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace
{

void require(bool condition, const char* message)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

} // namespace

int main()
{
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "seedvr2-invalid-video.avi";
    {
        std::FILE* file = std::fopen(path.string().c_str(), "wb");
        require(file != nullptr, "create malformed AVI");
        const char bytes[] = "not an avi";
        require(std::fwrite(bytes, 1, sizeof(bytes) - 1, file) == sizeof(bytes) - 1, "write malformed AVI");
        std::fclose(file);
    }

    seedvr2::AviVideoReader reader;
    std::string error;
    require(!seedvr2::AviVideoReader::open(path, reader, error), "reject malformed AVI");
    require(error.find("AVI") != std::string::npos, "malformed AVI error");
    std::remove(path.string().c_str());

#if !defined(SEEDVR2_HAS_FFMPEG)
    seedvr2::VideoReader generic_reader;
    require(!seedvr2::VideoReader::open("sample.mp4", generic_reader, error),
            "reject compressed input without FFmpeg");
    require(error.find("SEEDVR2_ENABLE_FFMPEG") != std::string::npos,
            "explain FFmpeg build option");
#endif

    const std::filesystem::path roundtrip = std::filesystem::temp_directory_path() / "seedvr2-roundtrip.avi";
    seedvr2::VideoInfo info;
    info.width = 2;
    info.height = 2;
    info.fps_num = 24;
    info.fps_den = 1;
    seedvr2::AviVideoWriter writer;
    require(seedvr2::AviVideoWriter::open(roundtrip, info, writer, error), "open AVI writer");
    seedvr2::RgbImage first;
    first.width = 2;
    first.height = 2;
    first.pixels = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255};
    seedvr2::RgbImage second = first;
    second.pixels[0] = 12;
    require(writer.write_frame(first, error), "write first AVI frame");
    require(writer.write_frame(second, error), "write second AVI frame");
    require(writer.close(error), "close AVI writer");

    require(seedvr2::AviVideoReader::open(roundtrip, reader, error), "open AVI reader");
    require(reader.info().width == 2, "AVI width");
    require(reader.info().height == 2, "AVI height");
    require(reader.info().frame_count == 2, "AVI frame count");
    require(reader.info().fps_num == 24, "AVI frame rate numerator");
    require(reader.info().fps_den == 1, "AVI frame rate denominator");
    seedvr2::RgbImage decoded;
    require(reader.read_next(decoded, error), "read first AVI frame");
    require(decoded.pixels == first.pixels, "first AVI frame pixels");
    require(reader.read_next(decoded, error), "read second AVI frame");
    require(decoded.pixels == second.pixels, "second AVI frame pixels");
    require(!reader.read_next(decoded, error), "AVI end of stream");

    seedvr2::VideoReader facade_reader;
    require(seedvr2::VideoReader::open(roundtrip, facade_reader, error), "open AVI through video facade");
    require(facade_reader.info().width == 2, "facade AVI width");
    require(facade_reader.read_next(decoded, error), "read AVI through video facade");
    require(decoded.pixels == first.pixels, "facade AVI pixels");
    std::remove(roundtrip.string().c_str());

#if defined(SEEDVR2_HAS_FFMPEG)
    const char* compressed_path = std::getenv("SEEDVR2_TEST_COMPRESSED_VIDEO");
    if (compressed_path && *compressed_path)
    {
        seedvr2::VideoReader compressed_reader;
        require(seedvr2::VideoReader::open(compressed_path, compressed_reader, error),
                "open compressed video through video facade");
        require(compressed_reader.info().width > 0, "compressed video width");
        require(compressed_reader.info().height > 0, "compressed video height");
        require(compressed_reader.info().fps_num > 0, "compressed video fps numerator");
        require(compressed_reader.info().fps_den > 0, "compressed video fps denominator");
        require(compressed_reader.read_next(decoded, error), "read first compressed video frame");
        require(decoded.pixels.size() == static_cast<std::size_t>(decoded.width) * decoded.height * 3u,
                "compressed video RGB24 frame");
        require(compressed_reader.read_next(decoded, error), "read second compressed video frame");
        require(!compressed_reader.read_next(decoded, error), "compressed video end of stream");
    }
#endif
    return 0;
}
