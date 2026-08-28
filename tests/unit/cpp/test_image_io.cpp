#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "io/image_io.h"

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
    const std::filesystem::path image_path =
        std::filesystem::temp_directory_path() / "seedvr2-ncnn-image-io-test.png";
    std::filesystem::remove(image_path);

    seedvr2::RgbImage source;
    source.width = 2;
    source.height = 1;
    source.pixels = {255, 0, 0, 0, 255, 0};
    std::string error;
    require(seedvr2::save_rgb_image(image_path, source, error), error.c_str());

    seedvr2::RgbImage decoded;
    require(seedvr2::load_rgb_image(image_path, decoded, error), error.c_str());
    require(decoded.width == 2 && decoded.height == 1, "decoded dimensions");
    require(decoded.pixels == source.pixels, "decoded RGB pixels");

    const std::filesystem::path jpeg_path =
        std::filesystem::temp_directory_path() / "seedvr2-ncnn-image-io-test.jpg";
    std::filesystem::remove(jpeg_path);
    require(seedvr2::save_rgb_image(jpeg_path, source, error), error.c_str());
    require(seedvr2::load_rgb_image(jpeg_path, decoded, error), error.c_str());
    require(decoded.width == 2 && decoded.height == 1 && decoded.pixels.size() == source.pixels.size(),
            "decoded JPEG dimensions");

    const std::filesystem::path missing_path = image_path.string() + ".missing";
    require(!seedvr2::load_rgb_image(missing_path, decoded, error), "missing image rejected");
    require(!error.empty(), "missing image error");

    const std::filesystem::path unsupported_path = image_path.string() + ".webp";
    require(!seedvr2::save_rgb_image(unsupported_path, source, error), "unsupported output rejected");
    require(error.find("unsupported") != std::string::npos, "unsupported output error");

    std::filesystem::remove(image_path);
    std::filesystem::remove(jpeg_path);
    std::puts("seedvr2-image-io: ok");
    return 0;
}
