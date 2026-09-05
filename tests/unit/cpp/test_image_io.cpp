#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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

    const std::filesystem::path input_dir =
        std::filesystem::temp_directory_path() / "seedvr2-ncnn-image-directory-test";
    std::filesystem::remove_all(input_dir);
    std::filesystem::create_directories(input_dir);
    std::ofstream(input_dir / "z.png") << "placeholder";
    std::ofstream(input_dir / "a.jpg") << "placeholder";
    std::ofstream(input_dir / "ignore.txt") << "placeholder";
    std::vector<std::filesystem::path> image_paths;
    require(seedvr2::list_rgb_images(input_dir, image_paths, error), error.c_str());
    require(image_paths == std::vector<std::filesystem::path>{input_dir / "a.jpg", input_dir / "z.png"},
            "directory image enumeration is sorted and filtered");

    std::vector<std::filesystem::path> output_paths;
    require(seedvr2::make_directory_output_paths(input_dir / "out", image_paths, output_paths, error),
            error.c_str());
    require(output_paths == std::vector<std::filesystem::path>{input_dir / "out" / "a.png",
                                                               input_dir / "out" / "z.png"},
            "directory output names follow input stems");
    const std::filesystem::path collision_jpeg = input_dir / "same.jpg";
    const std::filesystem::path collision_png = input_dir / "same.png";
    std::ofstream(collision_jpeg) << "placeholder";
    std::ofstream(collision_png) << "placeholder";
    std::vector<std::filesystem::path> collision_paths;
    require(seedvr2::list_rgb_images(input_dir, collision_paths, error), error.c_str());
    require(!seedvr2::make_directory_output_paths(input_dir / "out", collision_paths, output_paths, error),
            "directory output stem collision rejected");
    require(error.find("collision") != std::string::npos, "directory output collision error");
    std::filesystem::remove_all(input_dir);

    std::filesystem::remove(image_path);
    std::filesystem::remove(jpeg_path);
    std::puts("seedvr2-image-io: ok");
    return 0;
}
