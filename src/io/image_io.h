#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace seedvr2
{

struct RgbImage final
{
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;
};

bool load_rgb_image(const std::filesystem::path& path, RgbImage& image, std::string& error);
bool save_rgb_image(const std::filesystem::path& path, const RgbImage& image, std::string& error);

// Return supported image files in deterministic lexicographic order.
bool list_rgb_images(const std::filesystem::path& directory,
                     std::vector<std::filesystem::path>& paths,
                     std::string& error);

// Build the PNG destinations for a directory batch and reject stem collisions
// before any inference starts.
bool make_directory_output_paths(const std::filesystem::path& output_directory,
                                 const std::vector<std::filesystem::path>& input_paths,
                                 std::vector<std::filesystem::path>& output_paths,
                                 std::string& error);

} // namespace seedvr2
