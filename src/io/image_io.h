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

} // namespace seedvr2
